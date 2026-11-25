"""第一層：確定性純量化多週期基準預測模型（含波動率部位控管與機率校準）。"""

from __future__ import annotations

import math
from typing import Any

from twse_forecast.config import HORIZON_MAP, HORIZONS, TARGET_ANNUAL_VOLATILITY, HorizonSpec
from twse_forecast.models import HorizonQuantForecast


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _tanh_score(value: float | None, center: float, scale: float) -> float | None:
    if value is None or scale <= 0:
        return None
    return 100.0 * math.tanh((value - center) / scale)


def _weighted_score(components: dict[str, tuple[float | None, float]]) -> tuple[float, dict[str, float]]:
    """依有效因子動態正規化權重計算綜合分數（-100 ~ +100），單一因子缺值不致扭曲總分。"""
    total_w = 0.0
    weighted_sum = 0.0
    used_drivers: dict[str, float] = {}

    for name, (val, weight) in components.items():
        if val is None or weight <= 0:
            continue
        clamped = max(-100.0, min(100.0, val))
        total_w += weight
        weighted_sum += clamped * weight
        used_drivers[name] = round(clamped, 2)

    if total_w <= 0:
        return 0.0, used_drivers
    return round(weighted_sum / total_w, 2), used_drivers


def normalize_three_probs(p_up: float, p_flat: float, p_down: float) -> tuple[float, float, float]:
    """將多／盤／空機率安全正規化，並保證四捨五入至小數第 4 位後總和精確等於 1.0。"""
    vals = [max(0.0001, p_up), max(0.0001, p_flat), max(0.0001, p_down)]
    s = sum(vals)
    normed = [v / s for v in vals]
    rounded = [round(v, 4) for v in normed]
    diff = round(1.0 - sum(rounded), 4)
    if diff != 0.0:
        max_idx = max(range(3), key=lambda i: rounded[i])
        rounded[max_idx] = round(rounded[max_idx] + diff, 4)
    return rounded[0], rounded[1], rounded[2]


def score_to_calibrated_forecast(
    spec: HorizonSpec,
    score: float,
    latest_close: float,
    hist_mean: float,
    hist_std: float,
    key_drivers: dict[str, float],
) -> HorizonQuantForecast:
    """將單一週期量化分數透過貝氏收縮與常態累積分布換算為機率與 80% 信賴點位區間。"""
    clamped_score = max(-100.0, min(100.0, score))
    sigma = max(0.015, hist_std)
    # 貝氏收縮：即使滿分 100，也限制偏離歷史經驗均值最多 0.65 個標準差，避免過度自信
    expected_ret = hist_mean + (clamped_score / 100.0) * (0.65 * sigma)

    z_up = (spec.flat_threshold - expected_ret) / sigma
    z_down = (-spec.flat_threshold - expected_ret) / sigma

    raw_down = _norm_cdf(z_down)
    raw_up = 1.0 - _norm_cdf(z_up)
    raw_flat = max(0.0, _norm_cdf(z_up) - _norm_cdf(z_down))

    p_up, p_flat, p_down = normalize_three_probs(raw_up, raw_flat, raw_down)

    # 80% 雙尾信賴區間（z = 1.28155）
    z80 = 1.28155
    ci_low_ret = expected_ret - z80 * sigma
    ci_high_ret = expected_ret + z80 * sigma
    ci_low_lvl = latest_close * (1.0 + ci_low_ret)
    ci_high_lvl = latest_close * (1.0 + ci_high_ret)

    return HorizonQuantForecast(
        horizon_key=spec.key,
        label=spec.label,
        anchor_trading_days=spec.anchor_trading_days,
        quant_score=round(clamped_score, 2),
        expected_return=round(expected_ret, 5),
        std_dev=round(sigma, 5),
        prob_up=p_up,
        prob_flat=p_flat,
        prob_down=p_down,
        ci80_low_return=round(ci_low_ret, 5),
        ci80_high_return=round(ci_high_ret, 5),
        ci80_low_level=round(ci_low_lvl, 2),
        ci80_high_level=round(ci_high_lvl, 2),
        key_drivers=key_drivers,
    )


def compute_volatility_managed_allocation(
    scores_by_horizon: dict[str, float],
    realized_vol_20d: float | None,
) -> float:
    """依 Barroso & Santa-Clara (2015) 波動率倒數調節原則計算模擬持股水位（0.0 ~ 1.0）。"""
    weights = {
        "1_4w": 0.15,
        "1_3m": 0.30,
        "3_6m": 0.25,
        "6_12m": 0.20,
        "1_3y": 0.10,
    }
    composite = sum(scores_by_horizon.get(k, 0.0) * w for k, w in weights.items())
    # 基礎水位線性映射於 0.20 ~ 0.95（中性分數 0 對應約 58% 核心市場曝險）
    base_alloc = max(0.15, min(0.95, 0.58 + (composite / 100.0) * 0.40))

    vol = max(0.10, realized_vol_20d or TARGET_ANNUAL_VOLATILITY)
    vol_scaler = min(1.0, TARGET_ANNUAL_VOLATILITY / vol)
    return round(max(0.0, min(1.0, base_alloc * vol_scaler)), 4)


def run_quant_baseline(features: dict[str, Any]) -> dict[str, Any]:
    """對五個預測週期執行純量化基準評分與機率計算。"""
    latest_close = float(features["reference_taiex_close"])
    cross = features.get("cross_market") or {}
    fund = features.get("fundamentals_valuation") or {}
    micro = features.get("microstructure_flows") or {}
    tech = features.get("technicals") or {}
    breadth = features.get("breadth") or {}
    sentiment = features.get("sentiment") or {}
    hist_stats = features.get("historical_horizon_stats") or {}

    ma60_ded = tech.get("ma60_deduction") or {}
    deduct_4w_cushion = ma60_ded.get("deduction_cushion_4w_pct")

    # 共用單因子運算
    sox_5d = _tanh_score(cross.get("sox_ret_5d_pct"), 0.5, 4.0)
    sox_20d = _tanh_score(cross.get("sox_ret_20d_pct"), 1.0, 8.0)
    adr_prem = _tanh_score(cross.get("tsmc_adr_premium_pct"), 10.0, 10.0)
    twd_20d = _tanh_score(
        -cross["usd_twd_ret_20d_pct"] if cross.get("usd_twd_ret_20d_pct") is not None else None,
        0.0,
        1.8,
    )
    vix_score = _tanh_score(
        20.0 - cross["vix_latest"] if cross.get("vix_latest") is not None else None,
        0.0,
        8.0,
    )

    foreign_5d = _tanh_score(micro.get("foreign_5d_net_b"), 0.0, 450.0)
    trust_5d = _tanh_score(micro.get("trust_5d_net_b"), 0.0, 120.0)
    foreign_fut = _tanh_score(micro.get("foreign_tx_net_oi"), -40000.0, 35000.0)
    pc_ratio = _tanh_score(micro.get("pc_ratio_latest_pct"), 100.0, 22.0)

    # 短期乖離反轉（Jegadeesh 1990）：正乖離過高為負分、負乖離深反轉為正分
    bias_20d = tech.get("bias_20d_pct")
    short_rev_score = _tanh_score(-bias_20d if bias_20d is not None else None, 0.0, 4.5)
    ma60_ded_score = _tanh_score(deduct_4w_cushion, 0.0, 5.0)
    mom_12_1 = _tanh_score(tech.get("momentum_12_1m_pct"), 6.0, 22.0)

    # 市場寬度與拉積盤扣分
    pct_above_60 = breadth.get("pct_above_ma60")
    breadth_score = _tanh_score(pct_above_60, 50.0, 25.0)
    if breadth.get("narrow_leadership_warning") and breadth_score is not None:
        breadth_score = max(-100.0, breadth_score - 20.0)

    # 營收廣度與整體年增率
    rev_breadth = _tanh_score(fund.get("monthly_revenue_pos_ratio_pct"), 50.0, 15.0)
    rev_yoy = _tanh_score(fund.get("monthly_revenue_agg_yoy_pct"), 5.0, 14.0)

    # 估值與風險溢酬（台股中位數 P/B 約 1.6~1.9 為常態）
    pb_val = fund.get("market_median_pb")
    pb_score = _tanh_score(1.85 - pb_val if pb_val is not None else None, 0.0, 0.45)
    pe_val = fund.get("market_median_pe")
    pe_score = _tanh_score(17.5 - pe_val if pe_val is not None else None, 0.0, 5.0)
    erp_score = _tanh_score(fund.get("equity_risk_premium_pct"), -0.5, 1.8)

    # 總經殖利率利差與信用利差
    curve_score = _tanh_score(cross.get("yield_spread_pct"), 0.0, 0.9)
    hy_spread = cross.get("hy_credit_spread_pct")
    credit_score = _tanh_score(4.2 - hy_spread if hy_spread is not None else None, 0.0, 1.5)

    # 5 年價格位置（長線不宜在高檔 90% 過熱追高）
    pos_5y = tech.get("price_5y_percentile")
    pos_5y_anchor = _tanh_score(60.0 - pos_5y if pos_5y is not None else None, 0.0, 30.0)

    # PTT 散戶反向訊號
    ptt_contrarian = sentiment.get("contrarian_signal")

    # 每個週期的因子與權重定義
    horizon_inputs: dict[str, dict[str, tuple[float | None, float]]] = {
        "1_4w": {
            "跨市場費半與夜盤": (sox_5d, 0.16),
            "台積電ADR溢價支撐": (adr_prem, 0.10),
            "外資5日現貨與期貨籌碼": (
                (
                    (foreign_5d * 0.5 + foreign_fut * 0.5)
                    if foreign_5d is not None and foreign_fut is not None
                    else (foreign_5d if foreign_5d is not None else foreign_fut)
                ),
                0.22,
            ),
            "選擇權Put/Call籌碼": (pc_ratio, 0.12),
            "短期乖離均值反轉": (short_rev_score, 0.18),
            "季線4週扣抵方向": (ma60_ded_score, 0.10),
            "PTT股板反向情緒": (ptt_contrarian, 0.12),
        },
        "1_3m": {
            "上市月營收正成長廣度": (rev_breadth, 0.22),
            "外資與投信5日波段買盤": (
                (
                    (foreign_5d * 0.6 + trust_5d * 0.4)
                    if foreign_5d is not None and trust_5d is not None
                    else (foreign_5d or trust_5d)
                ),
                0.20,
            ),
            "季線扣抵與趨勢結構": (ma60_ded_score, 0.22),
            "代表籃子寬度與拉積濾網": (breadth_score, 0.16),
            "費半20日與台幣匯率趨勢": (
                (
                    (sox_20d * 0.6 + twd_20d * 0.4)
                    if sox_20d is not None and twd_20d is not None
                    else (sox_20d or twd_20d)
                ),
                0.12,
            ),
            "VIX波動率風險濾網": (vix_score, 0.08),
        },
        "3_6m": {
            "3–12個月價格動能": (mom_12_1, 0.30),
            "代表籃子市場寬度": (breadth_score, 0.22),
            "上市整體營收年增動能": (rev_yoy, 0.22),
            "季線扣抵中期支撐": (ma60_ded_score, 0.12),
            "信用利差與VIX總經環境": (
                (
                    (credit_score * 0.6 + vix_score * 0.4)
                    if credit_score is not None and vix_score is not None
                    else (credit_score or vix_score)
                ),
                0.14,
            ),
        },
        "6_12m": {
            "3–12個月價格動能": (mom_12_1, 0.26),
            "上市中位數P/B與P/E估值": (
                (
                    (pb_score * 0.6 + pe_score * 0.4)
                    if pb_score is not None and pe_score is not None
                    else (pb_score or pe_score)
                ),
                0.26,
            ),
            "美債殖利率利差與信用風險": (
                (
                    (curve_score * 0.5 + credit_score * 0.5)
                    if curve_score is not None and credit_score is not None
                    else (curve_score or credit_score)
                ),
                0.24,
            ),
            "上市月營收正成長廣度": (rev_breadth, 0.24),
        },
        "1_3y": {
            "上市中位數P/B長線估值錨": (pb_score, 0.35),
            "台股風險溢酬ERP": (erp_score, 0.30),
            "美債殖利率曲線與信用週期": (
                (
                    (curve_score * 0.6 + credit_score * 0.4)
                    if curve_score is not None and credit_score is not None
                    else (curve_score or credit_score)
                ),
                0.20,
            ),
            "5年位階均值回歸錨": (pos_5y_anchor, 0.15),
        },
    }

    forecasts: dict[str, HorizonQuantForecast] = {}
    scores: dict[str, float] = {}

    for spec in HORIZONS:
        score, drivers = _weighted_score(horizon_inputs[spec.key])
        scores[spec.key] = score
        h_stat = hist_stats.get(spec.key) or {}
        hist_mean = float(h_stat.get("mean_return", 0.02))
        hist_std = float(h_stat.get("std_return", 0.05))
        forecasts[spec.key] = score_to_calibrated_forecast(
            spec=spec,
            score=score,
            latest_close=latest_close,
            hist_mean=hist_mean,
            hist_std=hist_std,
            key_drivers=drivers,
        )

    target_alloc = compute_volatility_managed_allocation(
        scores_by_horizon=scores,
        realized_vol_20d=tech.get("realized_vol_20d"),
    )

    return {
        "reference_trade_date": features["reference_trade_date"],
        "reference_taiex_close": latest_close,
        "target_allocation": target_alloc,
        "horizons": forecasts,
    }
