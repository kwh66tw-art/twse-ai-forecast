"""彙整五大資料管線輸出為標準化週度特徵快照（FeatureSnapshot）。"""

from __future__ import annotations

from typing import Any

from twse_forecast.features.technicals import (
    compute_breadth_features,
    compute_historical_horizon_distributions,
    compute_taiex_technicals,
)
from twse_forecast.models import MarketSeries


def _series_return_pct(series_map: dict[str, MarketSeries], symbol: str, days: int) -> float | None:
    ser = series_map.get(symbol)
    if not ser or len(ser.bars) <= days:
        return None
    prev = ser.bars[-days - 1].close
    curr = ser.bars[-1].close
    if prev <= 0:
        return None
    return round((curr / prev - 1.0) * 100.0, 2)


def _latest_close(series_map: dict[str, MarketSeries], symbol: str) -> float | None:
    ser = series_map.get(symbol)
    if not ser or not ser.bars:
        return None
    return round(ser.bars[-1].close, 4)


def build_feature_snapshot(
    *,
    series_map: dict[str, MarketSeries],
    yahoo_summary: dict[str, Any],
    twse_payload: dict[str, Any],
    taifex_payload: dict[str, Any],
    fred_payload: dict[str, Any],
    sentiment_payload: dict[str, Any],
) -> dict[str, Any]:
    """建置可供純量化模型與 AI 辯證層共同讀取的特徵字典。"""
    taiex = series_map["^TWII"]
    etf0050 = series_map["0050.TW"]

    tech = compute_taiex_technicals(taiex)
    breadth = compute_breadth_features(series_map, tech["ret_20d_pct"])
    horizon_stats = compute_historical_horizon_distributions(taiex)

    # 跨市場與總經特徵（FRED 失敗時自動由 Yahoo 美債殖利率降級補位）
    fred_series = fred_payload.get("series") or {}
    us10y_latest = (
        fred_series["dgs10"]["latest"]
        if "dgs10" in fred_series
        else _latest_close(series_map, "^TNX")
    )
    us3m_yahoo = _latest_close(series_map, "^IRX")
    if "t10y2y" in fred_series:
        yield_spread = fred_series["t10y2y"]["latest"]
        yield_spread_source = "FRED:T10Y2Y"
    elif us10y_latest is not None and us3m_yahoo is not None:
        yield_spread = round(us10y_latest - us3m_yahoo, 3)
        yield_spread_source = "Yahoo:10Y-3M(降級替代)"
    else:
        yield_spread = None
        yield_spread_source = "unavailable"

    hy_oas = fred_series["hy_oas"]["latest"] if "hy_oas" in fred_series else None
    hy_oas_chg_20 = fred_series["hy_oas"]["change_20obs"] if "hy_oas" in fred_series else None

    cross_market = {
        "tsmc_adr_premium_pct": yahoo_summary.get("tsmc_adr_premium_pct"),
        "usd_twd_latest": yahoo_summary.get("usd_twd_latest"),
        "usd_twd_ret_20d_pct": _series_return_pct(series_map, "TWD=X", 20),
        "sox_ret_5d_pct": _series_return_pct(series_map, "^SOX", 5),
        "sox_ret_20d_pct": _series_return_pct(series_map, "^SOX", 20),
        "nvda_ret_5d_pct": _series_return_pct(series_map, "NVDA", 5),
        "sp500_ret_5d_pct": _series_return_pct(series_map, "^GSPC", 5),
        "nasdaq_ret_5d_pct": _series_return_pct(series_map, "^IXIC", 5),
        "sp_fut_ret_1d_pct": _series_return_pct(series_map, "ES=F", 1),
        "vix_latest": _latest_close(series_map, "^VIX"),
        "us10y_yield_pct": us10y_latest,
        "yield_spread_pct": yield_spread,
        "yield_spread_source": yield_spread_source,
        "hy_credit_spread_pct": hy_oas,
        "hy_credit_spread_chg_20d": hy_oas_chg_20,
    }

    # 基本面估值與台股風險溢酬 ERP（大盤殖利率 - 美債 10 年殖利率）
    twse_data = twse_payload.get("data") or {}
    val = twse_data.get("valuation") or {}
    rev = twse_data.get("monthly_revenue") or {}
    div_yield = val.get("median_dividend_yield_pct")
    erp_pct = (
        round(div_yield - us10y_latest, 2)
        if div_yield is not None and us10y_latest is not None
        else None
    )

    fundamentals = {
        "market_median_pb": val.get("median_pb"),
        "market_median_pe": val.get("median_pe"),
        "market_median_div_yield_pct": div_yield,
        "equity_risk_premium_pct": erp_pct,
        "monthly_revenue_ym": rev.get("data_year_month"),
        "monthly_revenue_pos_ratio_pct": rev.get("positive_yoy_ratio_pct"),
        "monthly_revenue_agg_yoy_pct": rev.get("aggregate_yoy_pct"),
        "monthly_revenue_median_yoy_pct": rev.get("median_yoy_pct"),
    }

    # 籌碼與微結構
    taifex_data = taifex_payload.get("data") or {}
    pc_ratio = taifex_data.get("put_call_ratio") or {}
    fut_oi = taifex_data.get("foreign_futures") or {}
    inst = twse_data.get("institutional_flow") or {}
    margin = twse_data.get("margin") or {}
    twse_breadth = twse_data.get("market_breadth_counts") or {}

    microstructure = {
        "foreign_5d_net_b": inst.get("sum_5d_foreign_net_b"),
        "trust_5d_net_b": inst.get("sum_5d_trust_net_b"),
        "dealer_prop_5d_net_b": inst.get("sum_5d_dealer_prop_net_b"),
        "foreign_1d_net_b": (inst.get("latest_1d") or {}).get("foreign_net_b"),
        "foreign_tx_net_oi": fut_oi.get("foreign_tx_net_oi"),
        "pc_ratio_latest_pct": pc_ratio.get("latest_pc_ratio_pct"),
        "pc_ratio_5d_avg_pct": pc_ratio.get("avg_5d_pc_ratio_pct"),
        "pc_ratio_20d_avg_pct": pc_ratio.get("avg_20d_pc_ratio_pct"),
        "pc_ratio_20d_diff": pc_ratio.get("pc_ratio_20d_diff"),
        "margin_balance_b": margin.get("margin_balance_b"),
        "margin_change_1d_pct": margin.get("margin_change_1d_pct"),
        "twse_advancing_count": twse_breadth.get("advancing"),
        "twse_declining_count": twse_breadth.get("declining"),
    }

    return {
        "reference_trade_date": taiex.bars[-1].date,
        "reference_taiex_close": round(taiex.bars[-1].close, 2),
        "reference_0050_trade_date": etf0050.bars[-1].date,
        "reference_0050_adj_close": round(etf0050.bars[-1].adj_close, 4),
        "cross_market": cross_market,
        "fundamentals_valuation": fundamentals,
        "microstructure_flows": microstructure,
        "technicals": tech,
        "breadth": breadth,
        "sentiment": sentiment_payload.get("ptt_stock"),
        "news_headlines": sentiment_payload.get("news_headlines") or [],
        "event_calendar": sentiment_payload.get("event_calendar") or [],
        "historical_horizon_stats": horizon_stats,
    }
