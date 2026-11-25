"""第一層：確定性純量化多週期基準預測模型（含貝氏漂移收縮、條件波動率融合與機率校準）。"""

from __future__ import annotations

import math
from typing import Any

from twse_forecast.config import HORIZONS, TARGET_ANNUAL_VOLATILITY, HorizonSpec
from twse_forecast.models import HorizonQuantForecast

# 台股長期均衡年化名目報酬率錨點（避免將短期 5 年大多頭線性外推為永久高報酬）
EQUILIBRIUM_ANNUAL_DRIFT = 0.085

# 肥尾先驗平滑係數（將極端常態尾端機率向 1/3 均勻先驗微幅收縮 8%，防止單次黑天鵝重創 Brier Score）
FAT_TAIL_PRIOR_WEIGHT = 0.08

# 各週期對「當前 20 日實現波動率」與「5 年歷史無條件波動率」之期限結構融合權重
REALIZED_VOL_TERM_WEIGHTS: dict[str, float] = {
    "1_4w": 0.55,
    "1_3m": 0.40,
    "3_6m": 0.25,
    "6_12m": 0.15,
    "1_3y": 0.05,
}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _tanh_score(value: float | None, center: float, scale: float) -> float | None:
    if value is None or scale <= 0:
        return None
    return 100.0 * math.tanh((value - center) / scale)


def _blend_optional(
    a: float | None,
    b: float | None,
    wa: float = 0.5,
    wb: float = 0.5,
) -> float | None:
    """安全合成兩個可能缺漏之因子分數（嚴格檢查 is not None，避免 0.0 被當成空值丟棄）。"""
    if a is not None and b is not None:
        total = wa + wb
        return (a * wa + b * wb) / total if total > 0 else 0.0
    if a is not None:
        return a
    return b


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


def shrink_horizon_drift(anchor_trading_days: int, sample_mean_return: float) -> float:
    """Bayes-Stein 漂移收縮：75% 錨定長期均衡年化報酬（8.5%），25% 參考有界歷史樣本均值。"""
    years = anchor_trading_days / 252.0
    eq_mean = (1.0 + EQUILIBRIUM_ANNUAL_DRIFT) ** years - 1.0
    if years <= 0:
        return 0.0
    # 將 5 年樣本隱含年化報酬限制在 [-5%, +12%]，防止單輪牛市估值擴張被當成常態漂移
    min_bound = (1.0 - 0.05) ** years - 1.0
    max_bound = (1.0 + 0.12) ** years - 1.0
    clamped_sample = max(min_bound, min(max_bound, sample_mean_return))
    return 0.75 * eq_mean + 0.25 * clamped_sample


def blend_conditional_volatility(
    spec: HorizonSpec,
    hist_std: float,
    realized_vol_20d: float | None = None,
    vol_multiplier: float = 1.0,
) -> float:
    """融合當前 20 日實現波動率與 5 年無條件標準差，並套用閉環經驗校準乘數。"""
    uncond_std = max(0.015, hist_std)
    w_curr = REALIZED_VOL_TERM_WEIGHTS.get(spec.key, 0.20)
    if realized_vol_20d is not None and realized_vol_20d > 0:
        implied_h_std = realized_vol_20d * math.sqrt(spec.anchor_trading_days / 252.0)
        # 限制短期極端跳空最多放大無條件標準差 1.85 倍、最低 0.65 倍
        capped_curr_std = max(uncond_std * 0.65, min(uncond_std * 1.85, implied_h_std))
        blended = w_curr * capped_curr_std + (1.0 - w_curr) * uncond_std
    else:
        blended = uncond_std

    safe_mult = max(0.85, min(1.35, vol_multiplier))
    return max(0.015, blended * safe_mult)


def score_to_calibrated_forecast(
    spec: HorizonSpec,
    score: float,
    latest_close: float,
    hist_mean: float,
    hist_std: float,
    key_drivers: dict[str, float],
    *,
    realized_vol_20d: float | None = None,
    vol_multiplier: float = 1.0,
) -> HorizonQuantForecast:
    """將單一週期分數透過貝氏漂移收縮、條件波動率與肥尾平滑換算為校準機率與 80% 區間。"""
    clamped_score = max(-100.0, min(100.0, score))
    base_drift = shrink_horizon_drift(spec.anchor_trading_days, hist_mean)
    sigma = blend_conditional_volatility(
        spec, hist_std, realized_vol_20d=realized_vol_20d, vol_multiplier=vol_multiplier
    )

    # 貝氏訊號收縮：滿分 ±100 最多偏離基準漂移 ±0.55 個條件標準差
    expected_ret = base_drift + (clamped_score / 100.0) * (0.55 * sigma)

    z_up = (spec.flat_threshold - expected_ret) / sigma
    z_down = (-spec.flat_threshold - expected_ret) / sigma

    raw_down = _norm_cdf(z_down)
    raw_up = 1.0 - _norm_cdf(z_up)
    raw_flat = max(0.0, _norm_cdf(z_up) - _norm_cdf(z_down))

    # 肥尾先驗平滑（8% 均勻先驗），降低黑天鵝事件下的 Brier Score 極端懲罰
    w_p = FAT_TAIL_PRIOR_WEIGHT
    smooth_up = (1.0 - w_p) * raw_up + w_p * (1.0 / 3.0)
    smooth_flat = (1.0 - w_p) * raw_flat + w_p * (1.0 / 3.0)
    smooth_down = (1.0 - w_p) * raw_down + w_p * (1.0 / 3.0)

    p_up, p_flat, p_down = normalize_three_probs(smooth_up, smooth_flat, smooth_down)

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
    base_alloc = max(0.15, min(0.95, 0.58 + (composite / 100.0) * 0.40))

    vol = max(0.10, realized_vol_20d or TARGET_ANNUAL_VOLATILITY)
    vol_scaler = min(1.0, TARGET_ANNUAL_VOLATILITY / vol)
    return round(max(0.0, min(1.0, base_alloc * vol_scaler)), 4)


def run_quant_baseline(
    features: dict[str, Any],
    *,
    calibration_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """對五個預測週期執行純量化基準評分與機率計算。"""
    latest_close = float(features["reference_taiex_close"])
    cross = features.get("cross_market") or {}
    fund = features.get("fundamentals_valuation") or {}
    micro = features.get("microstructure_flows") or {}
    tech = features.get("technicals") or {}
    breadth = features.get("breadth") or {}
    sentiment = features.get("sentiment") or {}
    hist_stats = features.get("historical_horizon_stats") or {}
    cal_by_h = (calibration_profile or {}).get("by_horizon") or {}

    ma60_ded = tech.get("ma60_deduction") or {}
    deduct_4w_cushion = ma60_ded.get("deduction_cushion_4w_pct")
    realized_vol_20d = tech.get("realized_vol_20d")

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

    # 選擇權 Put/Call Ratio：結合 60% 近 20 日邊際變化 + 40% 寬尺度絕對水位，避免結構性改變造成永久極端負分
    pc_diff_score = _tanh_score(micro.get("pc_ratio_20d_diff"), 0.0, 12.0)
    pc_level_score = _tanh_score(micro.get("pc_ratio_latest_pct"), 85.0, 35.0)
    pc_ratio = _blend_optional(pc_diff_score, pc_level_score, 0.60, 0.40)

    # 短期乖離反轉（Jegadeesh 1990）
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

    # 估值與風險溢酬
    pb_val = fund.get("market_median_pb")
    pb_score = _tanh_score(1.85 - pb_val if pb_val is not None else None, 0.0, 0.45)
    pe_val = fund.get("market_median_pe")
    pe_score = _tanh_score(17.5 - pe_val if pe_val is not None else None, 0.0, 5.0)
    erp_score = _tanh_score(fund.get("equity_risk_premium_pct"), -0.5, 1.8)

    # 總經殖利率利差與信用利差
    curve_score = _tanh_score(cross.get("yield_spread_pct"), 0.0, 0.9)
    hy_spread = cross.get("hy_credit_spread_pct")
    credit_score = _tanh_score(4.2 - hy_spread if hy_spread is not None else None, 0.0, 1.5)

    # 5 年價格位置
    pos_5y = tech.get("price_5y_percentile")
    pos_5y_anchor = _tanh_score(60.0 - pos_5y if pos_5y is not None else None, 0.0, 30.0)

    # PTT 散戶反向訊號
    ptt_contrarian = sentiment.get("contrarian_signal")

    # 每個週期的因子與權重定義（凍結固定，避免小樣本過度擬合）
    horizon_inputs: dict[str, dict[str, tuple[float | None, float]]] = {
        "1_4w": {
            "跨市場費半與夜盤": (sox_5d, 0.16),
            "台積電ADR溢價支撐": (adr_prem, 0.10),
            "外資5日現貨與期貨籌碼": (_blend_optional(foreign_5d, foreign_fut, 0.5, 0.5), 0.22),
            "選擇權Put/Call籌碼": (pc_ratio, 0.12),
            "短期乖離均值反轉": (short_rev_score, 0.18),
            "季線4週扣抵方向": (ma60_ded_score, 0.10),
            "PTT股板反向情緒": (ptt_contrarian, 0.12),
        },
        "1_3m": {
            "上市月營收正成長廣度": (rev_breadth, 0.22),
            "外資與投信5日波段買盤": (_blend_optional(foreign_5d, trust_5d, 0.6, 0.4), 0.20),
            "季線扣抵與趨勢結構": (ma60_ded_score, 0.22),
            "代表籃子寬度與拉積濾網": (breadth_score, 0.16),
            "費半20日與台幣匯率趨勢": (_blend_optional(sox_20d, twd_20d, 0.6, 0.4), 0.12),
            "VIX波動率風險濾網": (vix_score, 0.08),
        },
        "3_6m": {
            "3–12個月價格動能": (mom_12_1, 0.30),
            "代表籃子市場寬度": (breadth_score, 0.22),
            "上市整體營收年增動能": (rev_yoy, 0.22),
            "季線扣抵中期支撐": (ma60_ded_score, 0.12),
            "信用利差與VIX總經環境": (_blend_optional(credit_score, vix_score, 0.6, 0.4), 0.14),
        },
        "6_12m": {
            "3–12個月價格動能": (mom_12_1, 0.26),
            "上市中位數P/B與P/E估值": (_blend_optional(pb_score, pe_score, 0.6, 0.4), 0.26),
            "美債殖利率利差與信用風險": (_blend_optional(curve_score, credit_score, 0.5, 0.5), 0.24),
            "上市月營收正成長廣度": (rev_breadth, 0.24),
        },
        "1_3y": {
            "上市中位數P/B長線估值錨": (pb_score, 0.35),
            "台股風險溢酬ERP": (erp_score, 0.30),
            "美債殖利率曲線與信用週期": (_blend_optional(curve_score, credit_score, 0.6, 0.4), 0.20),
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
        vol_mult = float((cal_by_h.get(spec.key) or {}).get("vol_multiplier", 1.0))

        forecasts[spec.key] = score_to_calibrated_forecast(
            spec=spec,
            score=score,
            latest_close=latest_close,
            hist_mean=hist_mean,
            hist_std=hist_std,
            key_drivers=drivers,
            realized_vol_20d=realized_vol_20d,
            vol_multiplier=vol_mult,
        )

    target_alloc = compute_volatility_managed_allocation(
        scores_by_horizon=scores,
        realized_vol_20d=realized_vol_20d,
    )

    return {
        "reference_trade_date": features["reference_trade_date"],
        "reference_taiex_close": latest_close,
        "target_allocation": target_alloc,
        "horizons": forecasts,
    }
