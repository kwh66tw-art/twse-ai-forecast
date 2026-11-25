"""特徵工程（季線扣抵、拉積盤寬度）與量化基準／受限 AI 調整模型測試。"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from twse_forecast.config import HORIZONS, TAIWAN_BREADTH_BASKET
from twse_forecast.engine.briefing import validate_and_finalize_ai_submission
from twse_forecast.engine.quant_baseline import run_quant_baseline
from twse_forecast.features.feature_builder import build_feature_snapshot
from twse_forecast.features.technicals import compute_breadth_features, compute_ma60_deduction
from twse_forecast.models import MarketSeries, OHLCVBar, to_plain_dict


def _make_series(symbol: str, closes: list[float]) -> MarketSeries:
    start = date(2025, 1, 2)
    bars = []
    for i, c in enumerate(closes):
        d = (start + timedelta(days=i)).isoformat()
        bars.append(OHLCVBar(date=d, open=c, high=c * 1.01, low=c * 0.99, close=c, adj_close=c, volume=1000.0))
    return MarketSeries(symbol=symbol, bars=bars)


def test_ma60_deduction_regimes() -> None:
    # 情況一：前 40 天價格低 (100)，近 20 天上漲到 130 -> 未來 4 週扣抵低價區 100，具助漲支撐
    low_deduct = [100.0] * 45 + [110.0 + i for i in range(20)]
    res_up = compute_ma60_deduction(low_deduct)
    assert res_up["available"] is True
    assert res_up["deduction_cushion_4w_pct"] > 10.0
    assert "助漲" in res_up["regime"]

    # 情況二：前段價格高 (150)，目前跌至 110 -> 未來扣抵高價區 150，具下彎反壓
    high_deduct = [150.0] * 45 + [110.0] * 20
    res_dn = compute_ma60_deduction(high_deduct)
    assert res_dn["deduction_cushion_4w_pct"] < -10.0
    assert "反壓" in res_dn["regime"]


def test_breadth_narrow_leadership_warning() -> None:
    # 模擬大盤 20 日漲 +8%，但 30 檔代表籃子多數跌破季線且 20 日報酬 -2%（典型拉積盤背離）
    series_map: dict[str, MarketSeries] = {}
    down_closes = [120.0] * 45 + [102.0 - i * 0.2 for i in range(20)]
    for sym in TAIWAN_BREADTH_BASKET:
        series_map[sym] = _make_series(sym, down_closes)

    breadth = compute_breadth_features(series_map, taiex_ret_20d_pct=8.0)
    assert breadth["available"] is True
    assert breadth["pct_above_ma60"] == 0.0
    assert breadth["taiex_vs_equal_spread_20d_pct"] > 5.0
    assert breadth["narrow_leadership_warning"] is True


def test_quant_probs_sum_to_one_and_bounded_ai_override() -> None:
    # 建立 300 筆模擬日線
    taiex_closes = [20000.0 + i * 12.0 + (i % 5) * 30.0 for i in range(300)]
    etf_closes = [180.0 + i * 0.12 for i in range(300)]
    series_map = {
        "^TWII": _make_series("^TWII", taiex_closes),
        "0050.TW": _make_series("0050.TW", etf_closes),
        "2330.TW": _make_series("2330.TW", [900.0 + i * 0.5 for i in range(300)]),
        "TSM": _make_series("TSM", [160.0 + i * 0.1 for i in range(300)]),
        "TWD=X": _make_series("TWD=X", [32.0 for _ in range(300)]),
        "^SOX": _make_series("^SOX", [4800.0 + i * 2.0 for i in range(300)]),
        "^VIX": _make_series("^VIX", [15.0 for _ in range(300)]),
        "^TNX": _make_series("^TNX", [4.2 for _ in range(300)]),
        "^IRX": _make_series("^IRX", [4.0 for _ in range(300)]),
    }
    for sym in TAIWAN_BREADTH_BASKET[:10]:
        series_map.setdefault(sym, _make_series(sym, [100.0 + i * 0.1 for i in range(300)]))

    features = build_feature_snapshot(
        series_map=series_map,
        yahoo_summary={"tsmc_adr_premium_pct": 12.0, "usd_twd_latest": 32.0},
        twse_payload={"data": {"valuation": {"median_pb": 1.75, "median_pe": 16.0, "median_dividend_yield_pct": 3.8}, "monthly_revenue": {"positive_yoy_ratio_pct": 62.0, "aggregate_yoy_pct": 11.0}, "institutional_flow": {"sum_5d_foreign_net_b": 250.0, "sum_5d_trust_net_b": 80.0}}},
        taifex_payload={"data": {"put_call_ratio": {"latest_pc_ratio_pct": 115.0}, "foreign_futures": {"foreign_tx_net_oi": -15000}}},
        fred_payload={"series": {"dgs10": {"latest": 4.2}, "t10y2y": {"latest": 0.25}, "hy_oas": {"latest": 3.2, "change_20obs": -0.1}}},
        sentiment_payload={"ptt_stock": {"contrarian_signal": 0.0}, "news_headlines": [], "event_calendar": []},
    )
    quant = to_plain_dict(run_quant_baseline(features))
    for spec in HORIZONS:
        qh = quant["horizons"][spec.key]
        total_p = qh["prob_up"] + qh["prob_flat"] + qh["prob_down"]
        assert total_p == pytest.approx(1.0, abs=1e-4)

    # 測試 AI 刻意提交超標 +80 分的調整，系統必須強制截斷為 +25 分內
    raw_ai = {
        "model_name": "Test AI",
        "bull_case": "基本面與營收廣度佳",
        "bear_case": "地緣與估值風險仍存",
        "cio_synthesis": "偏多看待",
        "override_rationale": "測試邊界截斷",
        "horizons": {
            spec.key: {
                "ai_score": quant["horizons"][spec.key]["quant_score"] + 80.0,
                "thesis": f"{spec.label} 論述",
            }
            for spec in HORIZONS
        },
    }
    submission = validate_and_finalize_ai_submission(raw_ai, quant, features)
    for spec in HORIZONS:
        ah = submission.horizons[spec.key]
        assert abs(ah.delta_score) <= 25.0 + 1e-6
        assert ah.prob_up + ah.prob_flat + ah.prob_down == pytest.approx(1.0, abs=1e-4)
