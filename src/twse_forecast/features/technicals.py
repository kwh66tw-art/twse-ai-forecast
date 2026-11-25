"""多週期 K 線技術指標、季線扣抵預判、學術動能／反轉因子與拉積盤寬度計算。"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from twse_forecast.config import TAIWAN_BREADTH_BASKET
from twse_forecast.models import MarketSeries


def sma(values: list[float], window: int) -> float | None:
    if len(values) < window or window <= 0:
        return None
    return sum(values[-window:]) / window


def sma_series(values: list[float], window: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if window <= 0 or len(values) < window:
        return out
    running = sum(values[:window])
    out[window - 1] = running / window
    for i in range(window, len(values)):
        running += values[i] - values[i - window]
        out[i] = running / window
    return out


def compute_rsi(values: list[float], period: int = 14) -> float | None:
    """Wilder 平滑法計算相對強弱指標 RSI(14)。"""
    if len(values) <= period:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, period + 1):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    for i in range(period + 1, len(values)):
        diff = values[i] - values[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(diff, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-diff, 0.0)) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_atr_pct(series: MarketSeries, period: int = 14) -> float | None:
    """計算真實波動幅度均值 ATR(14) 占最新收盤價百分比。"""
    bars = series.bars
    if len(bars) <= period:
        return None
    trs: list[float] = []
    for i in range(1, len(bars)):
        prev_c = bars[i - 1].close
        tr = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - prev_c),
            abs(bars[i].low - prev_c),
        )
        trs.append(tr)
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    latest_close = bars[-1].close
    if latest_close <= 0:
        return None
    return (atr / latest_close) * 100.0


def compute_realized_vol_annualized(closes: list[float], window: int = 20) -> float | None:
    """計算近 N 日對數報酬率年化波動度（小數，例如 0.18 代表 18%）。"""
    if len(closes) <= window:
        return None
    rets: list[float] = []
    sub = closes[-(window + 1) :]
    for i in range(1, len(sub)):
        if sub[i - 1] > 0 and sub[i] > 0:
            rets.append(math.log(sub[i] / sub[i - 1]))
    if len(rets) < 5:
        return None
    mean_r = sum(rets) / len(rets)
    var = sum((r - mean_r) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252.0)


def compute_bollinger_squeeze_pct(closes: list[float], window: int = 20, lookback: int = 120) -> float | None:
    """計算最新布林通道頻寬在過去 120 日內所處的百分位（越低代表越緊縮）。"""
    if len(closes) < window + 10:
        return None
    widths: list[float] = []
    start = max(window - 1, len(closes) - lookback)
    for idx in range(start, len(closes)):
        seg = closes[idx - window + 1 : idx + 1]
        mid = sum(seg) / window
        if mid <= 0:
            continue
        std = math.sqrt(sum((x - mid) ** 2 for x in seg) / window)
        widths.append((4.0 * std) / mid)
    if len(widths) < 5:
        return None
    curr_w = widths[-1]
    rank = sum(1 for w in widths if w <= curr_w)
    return (rank / len(widths)) * 100.0


def compute_ma60_deduction(closes: list[float]) -> dict[str, Any]:
    """預判季線（60MA）未來 1 週與 4 週的扣抵壓力或支撐。"""
    if len(closes) < 60:
        return {"available": False}
    curr = closes[-1]
    ma60_now = sum(closes[-60:]) / 60.0
    ma60_prev5 = sum(closes[-65:-5]) / 60.0 if len(closes) >= 65 else ma60_now
    slope_5d_pct = ((ma60_now - ma60_prev5) / ma60_prev5 * 100.0) if ma60_prev5 > 0 else 0.0

    # 未來 5 日即將被移出平均線計算窗的是 closes[-60:-55]，第 5 日代表為 closes[-56]
    deduct_1w = closes[-56]
    # 未來 20 日即將移出的是 closes[-60:-40]，第 20 日代表為 closes[-41]
    deduct_4w = closes[-41]

    edge_1w_pct = (curr - deduct_1w) / deduct_1w * 100.0 if deduct_1w > 0 else 0.0
    edge_4w_pct = (curr - deduct_4w) / deduct_4w * 100.0 if deduct_4w > 0 else 0.0

    if edge_4w_pct > 2.0:
        regime = "扣抵低價區（季線具自然助漲支撐）"
    elif edge_4w_pct < -2.0:
        regime = "扣抵高價區（季線具自然下彎反壓）"
    else:
        regime = "扣抵平盤區（季線趨勢中性待變）"

    return {
        "available": True,
        "ma60": round(ma60_now, 2),
        "ma60_slope_5d_pct": round(slope_5d_pct, 3),
        "deduct_1w_ref_price": round(deduct_1w, 2),
        "deduct_4w_ref_price": round(deduct_4w, 2),
        "deduction_cushion_1w_pct": round(edge_1w_pct, 2),
        "deduction_cushion_4w_pct": round(edge_4w_pct, 2),
        "regime": regime,
    }


def _resample_monthly_closes(series: MarketSeries) -> list[float]:
    monthly: dict[str, float] = {}
    for bar in series.bars:
        ym = bar.date[:7]
        monthly[ym] = bar.close
    return [monthly[k] for k in sorted(monthly)]


def compute_taiex_technicals(series: MarketSeries) -> dict[str, Any]:
    """計算台灣加權指數的多週期 K 線技術結構與學術實證動能／反轉因子。"""
    closes = series.closes()
    if len(closes) < 60:
        raise ValueError("^TWII K 線筆數不足以計算多週期指標")

    curr = closes[-1]
    ma5 = sma(closes, 5)
    ma20 = sma(closes, 20)
    ma60 = sma(closes, 60)
    ma120 = sma(closes, 120)
    ma240 = sma(closes, 240)

    def pct_ret(days: int) -> float | None:
        if len(closes) <= days or closes[-days - 1] <= 0:
            return None
        return (curr / closes[-days - 1] - 1.0) * 100.0

    # Jegadeesh & Titman (1993) 3–12 個月價格動能（過去 252 日剔除最近 21 日）
    if len(closes) >= 253 and closes[-253] > 0:
        mom_12_1 = (closes[-22] / closes[-253] - 1.0) * 100.0
    elif len(closes) >= 126 and closes[-126] > 0:
        mom_12_1 = (closes[-22] / closes[-126] - 1.0) * 100.0
    else:
        mom_12_1 = pct_ret(60) or 0.0

    bias_20d_pct = ((curr / ma20) - 1.0) * 100.0 if ma20 else 0.0
    bias_60d_pct = ((curr / ma60) - 1.0) * 100.0 if ma60 else 0.0
    bias_240d_pct = ((curr / ma240) - 1.0) * 100.0 if ma240 else None

    monthly_closes = _resample_monthly_closes(series)
    ma12m = sma(monthly_closes, 12)
    ma36m = sma(monthly_closes, 36)

    # 歷史 5 年區間位置 (0~100%)
    high_5y = max(closes)
    low_5y = min(closes)
    pos_5y_pct = ((curr - low_5y) / (high_5y - low_5y) * 100.0) if high_5y > low_5y else 50.0

    return {
        "latest_date": series.bars[-1].date,
        "latest_close": round(curr, 2),
        "ret_5d_pct": round(pct_ret(5) or 0.0, 2),
        "ret_20d_pct": round(pct_ret(20) or 0.0, 2),
        "ret_60d_pct": round(pct_ret(60) or 0.0, 2),
        "ret_120d_pct": round(pct_ret(120), 2) if pct_ret(120) is not None else None,
        "ret_240d_pct": round(pct_ret(240), 2) if pct_ret(240) is not None else None,
        "momentum_12_1m_pct": round(mom_12_1, 2),
        "ma5": round(ma5, 2) if ma5 else None,
        "ma20": round(ma20, 2) if ma20 else None,
        "ma60": round(ma60, 2) if ma60 else None,
        "ma120": round(ma120, 2) if ma120 else None,
        "ma240": round(ma240, 2) if ma240 else None,
        "ma12m": round(ma12m, 2) if ma12m else None,
        "ma36m": round(ma36m, 2) if ma36m else None,
        "bias_20d_pct": round(bias_20d_pct, 2),
        "bias_60d_pct": round(bias_60d_pct, 2),
        "bias_240d_pct": round(bias_240d_pct, 2) if bias_240d_pct is not None else None,
        "rsi14": round(compute_rsi(closes, 14) or 50.0, 2),
        "atr14_pct": round(compute_atr_pct(series, 14) or 1.2, 3),
        "realized_vol_20d": round(compute_realized_vol_annualized(closes, 20) or 0.16, 4),
        "bollinger_squeeze_pct": round(compute_bollinger_squeeze_pct(closes) or 50.0, 1),
        "price_5y_percentile": round(pos_5y_pct, 1),
        "ma60_deduction": compute_ma60_deduction(closes),
    }


def compute_breadth_features(
    series_map: dict[str, MarketSeries],
    taiex_ret_20d_pct: float,
) -> dict[str, Any]:
    """計算 30 檔代表籃子之市場寬度（站上均線家數比與拉積盤背離指標）。"""
    above_20 = 0
    above_60 = 0
    above_240 = 0
    count_20 = 0
    count_60 = 0
    count_240 = 0
    rets_20d: list[float] = []

    for sym in TAIWAN_BREADTH_BASKET:
        ser = series_map.get(sym)
        if not ser or len(ser.bars) < 25:
            continue
        closes = ser.closes()
        curr = closes[-1]
        m20 = sma(closes, 20)
        if m20:
            count_20 += 1
            if curr > m20:
                above_20 += 1
        m60 = sma(closes, 60)
        if m60:
            count_60 += 1
            if curr > m60:
                above_60 += 1
        m240 = sma(closes, 240)
        if m240:
            count_240 += 1
            if curr > m240:
                above_240 += 1
        if len(closes) > 20 and closes[-21] > 0:
            rets_20d.append((curr / closes[-21] - 1.0) * 100.0)

    if count_20 == 0 or not rets_20d:
        return {"available": False}

    pct_above_20 = above_20 / count_20 * 100.0
    pct_above_60 = (above_60 / count_60 * 100.0) if count_60 else 50.0
    pct_above_240 = (above_240 / count_240 * 100.0) if count_240 else 50.0
    equal_weight_ret_20d = sum(rets_20d) / len(rets_20d)
    # 正值越大代表權值股獨漲而多數股票沒跟上（潛在拉積盤背離）
    cap_vs_equal_spread = taiex_ret_20d_pct - equal_weight_ret_20d

    narrow_leadership_warning = cap_vs_equal_spread > 3.0 and pct_above_60 < 50.0

    return {
        "available": True,
        "basket_size_used": count_20,
        "pct_above_ma20": round(pct_above_20, 1),
        "pct_above_ma60": round(pct_above_60, 1),
        "pct_above_ma240": round(pct_above_240, 1),
        "equal_weight_ret_20d_pct": round(equal_weight_ret_20d, 2),
        "taiex_vs_equal_spread_20d_pct": round(cap_vs_equal_spread, 2),
        "narrow_leadership_warning": narrow_leadership_warning,
    }


def compute_historical_horizon_distributions(series: MarketSeries) -> dict[str, dict[str, float]]:
    """從近 5 年真實 ^TWII 收盤序列無前視估算各週期的歷史經驗平均報酬率與標準差。"""
    from twse_forecast.config import HORIZONS

    closes = series.closes()
    stats: dict[str, dict[str, float]] = {}
    for h in HORIZONS:
        n = h.anchor_trading_days
        if len(closes) <= n + 20:
            # 若歷史長度不足則依日波動率與 √n 縮放估算
            daily_vol = (compute_realized_vol_annualized(closes, min(60, len(closes) - 1)) or 0.16) / math.sqrt(252.0)
            stats[h.key] = {
                "mean_return": round(0.08 * (n / 252.0), 5),
                "std_return": round(max(0.02, daily_vol * math.sqrt(n)), 5),
                "sample_count": 0,
            }
            continue
        fwd_rets = [
            (closes[i + n] / closes[i]) - 1.0
            for i in range(len(closes) - n)
            if closes[i] > 0
        ]
        mean_r = sum(fwd_rets) / len(fwd_rets)
        var_r = sum((r - mean_r) ** 2 for r in fwd_rets) / max(1, len(fwd_rets) - 1)
        std_r = max(0.015, math.sqrt(var_r))
        stats[h.key] = {
            "mean_return": round(mean_r, 5),
            "std_return": round(std_r, 5),
            "sample_count": len(fwd_rets),
        }
    return stats
