"""深度稽核修復與閉環自我校準模組單元測試：貝氏漂移收縮、條件波動率、盤中殘缺 K 剔除與動態授權韁繩。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from twse_forecast.collectors.market_yahoo import parse_yahoo_chart_payload
from twse_forecast.collectors.taifex_official import parse_taifex_pc_ratio
from twse_forecast.config import HORIZON_MAP, HORIZONS, TAIPEI_TZ
from twse_forecast.engine.briefing import (
    render_briefing_markdown,
    validate_and_finalize_ai_submission,
)
from twse_forecast.engine.gemini_agent import _build_prompts
from twse_forecast.engine.quant_baseline import (
    _blend_optional,
    blend_conditional_volatility,
    score_to_calibrated_forecast,
    shrink_horizon_drift,
)
from twse_forecast.evaluation.calibration import compute_calibration_profile


def test_blend_optional_preserves_exact_zero_float() -> None:
    """驗證 0.0 中性分數不會因為 Python truthiness (0.0 or b) 被誤當成缺值丟棄。"""
    assert _blend_optional(0.0, None, 0.6, 0.4) == pytest.approx(0.0)
    assert _blend_optional(None, 0.0, 0.6, 0.4) == pytest.approx(0.0)
    assert _blend_optional(0.0, -50.0, 0.5, 0.5) == pytest.approx(-25.0)
    assert _blend_optional(None, None, 0.5, 0.5) is None


def test_bayes_stein_drift_shrinkage_and_fat_tail_prior() -> None:
    """驗證當 5 年樣本呈現極端大多頭 (+60%) 時，漂移收縮與肥尾平滑能抑制極端機率過度自信。"""
    spec_1_3y = HORIZON_MAP["1_3y"]
    shrunk = shrink_horizon_drift(spec_1_3y.anchor_trading_days, sample_mean_return=0.60)
    # 375 交易日均衡約 +12.9%，樣本上限約 +18.4%，75/25 融合後約 14.3%，絕不外推至 +60%
    assert 0.12 <= shrunk <= 0.16

    fc = score_to_calibrated_forecast(
        spec=spec_1_3y,
        score=20.0,
        latest_close=40000.0,
        hist_mean=0.60,
        hist_std=0.22,
        key_drivers={},
        realized_vol_20d=0.35,
    )
    assert fc.prob_up + fc.prob_flat + fc.prob_down == pytest.approx(1.0, abs=1e-4)
    # 肥尾先驗平滑確保空頭尾端機率不會歸零，避免黑天鵝重創 Brier Score
    assert fc.prob_down >= 0.05
    assert fc.prob_up < 0.80


def test_conditional_volatility_term_structure_and_multiplier() -> None:
    """驗證高實現波動率 (60%) 對短週期標準差的放大顯著高於長週期，且受經驗乘數有界約束。"""
    spec_short = HORIZON_MAP["1_4w"]
    spec_long = HORIZON_MAP["1_3y"]

    low_vol_short = blend_conditional_volatility(spec_short, hist_std=0.06, realized_vol_20d=0.18)
    high_vol_short = blend_conditional_volatility(spec_short, hist_std=0.06, realized_vol_20d=0.60)
    assert high_vol_short > low_vol_short * 1.35

    low_vol_long = blend_conditional_volatility(spec_long, hist_std=0.25, realized_vol_20d=0.18)
    high_vol_long = blend_conditional_volatility(spec_long, hist_std=0.25, realized_vol_20d=0.60)
    # 1_3y 僅給 5% 近期實現波動率權重，保持長期區間穩定
    assert high_vol_long / low_vol_long < 1.06

    # 經驗乘數邊界保護 [0.85, 1.35]
    capped = blend_conditional_volatility(spec_short, hist_std=0.06, realized_vol_20d=None, vol_multiplier=9.0)
    assert capped == pytest.approx(0.06 * 1.35, rel=1e-4)


def test_intraday_incomplete_taiwan_bar_dropped_before_market_close() -> None:
    """驗證週一至週五 13:45 收盤前，Yahoo 台股序列自動剔除當日盤中未收盤 K 棒，但美股不受影響。"""
    # 2026-11-20 (五) 08:00 UTC -> 台北 16:00 (2026-11-20)
    # 2026-11-23 (一) 01:15 UTC -> 台北 09:15 (2026-11-23 盤中)
    ts_fri = int(datetime(2026, 11, 20, 13, 30, tzinfo=TAIPEI_TZ).timestamp())
    ts_mon_intraday = int(datetime(2026, 11, 23, 9, 15, tzinfo=TAIPEI_TZ).timestamp())
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [ts_fri, ts_mon_intraday],
                    "indicators": {
                        "quote": [
                            {
                                "open": [40000.0, 40100.0],
                                "high": [40200.0, 40150.0],
                                "low": [39900.0, 40050.0],
                                "close": [40122.0, 40080.0],
                                "volume": [1000, 120],
                            }
                        ]
                    },
                }
            ]
        }
    }
    monday_morning = datetime(2026, 11, 23, 9, 20, tzinfo=TAIPEI_TZ)
    monday_after_close = datetime(2026, 11, 23, 14, 0, tzinfo=TAIPEI_TZ)

    tw_intraday = parse_yahoo_chart_payload("^TWII", payload, now_taipei=monday_morning)
    assert len(tw_intraday.bars) == 1
    assert tw_intraday.bars[-1].date == "2026-11-20"

    tw_closed = parse_yahoo_chart_payload("^TWII", payload, now_taipei=monday_after_close)
    assert len(tw_closed.bars) == 2
    assert tw_closed.bars[-1].date == "2026-11-23"

    us_sym = parse_yahoo_chart_payload("^SOX", payload, now_taipei=monday_morning)
    assert len(us_sym.bars) == 2


def test_pc_ratio_marginal_20d_diff() -> None:
    rows = [{"Date": f"202611{d:02d}", "PutCallOIRatio%": "50.0"} for d in range(1, 20)]
    rows.append({"Date": "20261120", "PutCallOIRatio%": "70.0"})
    res = parse_taifex_pc_ratio(rows)
    assert res["avg_20d_pc_ratio_pct"] == pytest.approx(51.0)
    assert res["pc_ratio_20d_diff"] == pytest.approx(19.0)


def test_closed_loop_calibration_profile_and_dynamic_ai_leash() -> None:
    """驗證累積滿 6 筆結算後，若區間涵蓋過低且 AI 主觀調整落後量化，自動放大波動乘數並收緊授權上限至 ±12 分。"""
    records = [
        {
            "record_id": f"2026-W{w}",
            "ai_forecast": {
                "horizons": {
                    "1_4w": {
                        "delta_score": 20.0,
                        "prob_up": 0.72,
                        "prob_flat": 0.18,
                        "prob_down": 0.10,
                    }
                }
            },
        }
        for w in range(1, 8)
    ]

    # 先測 3 筆（未滿 6 筆門檻 -> 不縮放硬參數）
    few_settlements: list[dict[str, Any]] = [
        {
            "record_id": f"2026-W{w}",
            "horizon_key": "1_4w",
            "horizon_label": "1–4 週",
            "ai_expected_return": 0.04,
            "actual_return": -0.05,
            "actual_outcome": "DOWN",
            "ai_pred_class": "UP",
            "ai_hit": False,
            "in_ai_ci80": False,
            "ai_brier": 0.92,
            "quant_brier": 0.70,
        }
        for w in range(1, 4)
    ]
    prof_early = compute_calibration_profile(records, few_settlements)
    assert prof_early["by_horizon"]["1_4w"]["active"] is False
    assert prof_early["by_horizon"]["1_4w"]["vol_multiplier"] == 1.0
    assert prof_early["by_horizon"]["1_4w"]["max_ai_delta"] == 25.0

    # 滿 6 筆且持續高估、區間失準、AI Brier 劣於量化 -> 啟動波動率放大與 ±12 分緊縮韁繩
    full_settlements = [
        {
            "record_id": f"2026-W{w}",
            "horizon_key": "1_4w",
            "horizon_label": "1–4 週",
            "ai_expected_return": 0.04,
            "actual_return": -0.05,
            "actual_outcome": "DOWN",
            "ai_pred_class": "UP",
            "ai_hit": False,
            "in_ai_ci80": w == 1,  # 涵蓋率僅 1/6 = 16.7%
            "ai_brier": 0.90,
            "quant_brier": 0.72,  # ai_brier_value_add = -0.18 < -0.04
        }
        for w in range(1, 7)
    ]
    prof_active = compute_calibration_profile(records, full_settlements)
    h_cal = prof_active["by_horizon"]["1_4w"]
    assert h_cal["active"] is True
    assert h_cal["bias_direction"] == "over_optimistic"
    assert h_cal["vol_multiplier"] == 1.35
    assert h_cal["max_ai_delta"] == 12.0
    assert len(prof_active["high_confidence_misses"]) == 3

    # 驗證簡報與提示詞均正確注入 ±12 分限制與誤判警示
    quant = {
        "target_allocation": 0.5,
        "horizons": {
            spec.key: {
                "horizon_key": spec.key,
                "label": spec.label,
                "anchor_trading_days": spec.anchor_trading_days,
                "quant_score": 10.0,
                "expected_return": 0.01,
                "prob_up": 0.4,
                "prob_flat": 0.3,
                "prob_down": 0.3,
                "ci80_low_level": 38000.0,
                "ci80_high_level": 42000.0,
            }
            for spec in HORIZONS
        },
    }
    features = {
        "reference_trade_date": "2026-11-24",
        "reference_taiex_close": 40000.0,
        "reference_0050_trade_date": "2026-11-24",
        "reference_0050_adj_close": 95.0,
        "technicals": {"realized_vol_20d": 0.25},
        "historical_horizon_stats": {
            spec.key: {"mean_return": 0.02, "std_return": 0.05}
            for spec in HORIZONS
        },
    }
    briefing_md = render_briefing_markdown(
        record_id="2026-W55",
        features=features,
        quant_result=quant,
        recent_settled_reviews=full_settlements[-2:],
        collector_reports=[],
        calibration_profile=prof_active,
    )
    assert "滾動偏差與自我校準診斷表" in briefing_md
    assert "近期高信心誤判警示" in briefing_md

    _, user_prompt = _build_prompts(briefing_md, quant, calibration_profile=prof_active)
    assert "±12 分，合法 ai_score 範圍 [-2.00, +22.00]" in user_prompt

    # 驗證後端強制將 AI 提交的 +25 分截斷在動態韁繩 ±12 分內
    raw_ai = {
        "model_name": "gemini-4-neon",
        "bull_case": "B",
        "bear_case": "R",
        "cio_synthesis": "S",
        "override_rationale": "O",
        "horizons": {
            spec.key: {"ai_score": 35.0, "thesis": "測試動態韁繩"}
            for spec in HORIZONS
        },
    }
    finalized = validate_and_finalize_ai_submission(
        raw_ai, quant, features, calibration_profile=prof_active
    )
    assert finalized.horizons["1_4w"].delta_score == pytest.approx(12.0)
    # 未達門檻之其他週期仍保留預設 ±25 分上限
    assert finalized.horizons["1_3m"].delta_score == pytest.approx(25.0)
