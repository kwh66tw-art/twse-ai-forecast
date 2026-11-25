"""Yahoo Finance v8 Chart API 擷取器（含證交所官方歷史日線自動備援）。"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx

from twse_forecast.collectors.http_utils import fetch_with_retry, now_taipei_iso, parse_number
from twse_forecast.config import CORE_MARKET_SYMBOLS, TAIPEI_TZ, TAIWAN_BREADTH_BASKET
from twse_forecast.models import CollectorReport, MarketSeries, OHLCVBar

YAHOO_HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
# Yahoo Finance Chart API 對簡潔單純之 macOS User-Agent（不帶 Accept-Language 或 events=div,split）最穩定
YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

TWSE_TAIEX_HIST_URL = "https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST"
TWSE_STOCK_DAY_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"


def roc_to_iso_date(roc_date: str) -> str | None:
    """將台灣證交所民國日期字串（例如 '115/11/24'）轉為西元 '2026-11-24'。"""
    parts = roc_date.strip().split("/")
    if len(parts) != 3:
        return None
    try:
        year = int(parts[0]) + 1911
        month = int(parts[1])
        day = int(parts[2])
        return f"{year:04d}-{month:02d}-{day:02d}"
    except ValueError:
        return None


def parse_yahoo_chart_payload(
    symbol: str,
    payload: dict[str, Any],
    *,
    now_taipei: datetime | None = None,
) -> MarketSeries:
    """解析 Yahoo Finance v8 chart JSON 為標準 OHLCVBar 序列（自動剔除台股盤中未完成日 K）。"""
    chart = payload.get("chart") or {}
    results = chart.get("result") or []
    if not results:
        error = chart.get("error") or "empty result"
        raise ValueError(f"Yahoo Finance 回傳空資料 ({symbol}): {error}")

    item = results[0]
    timestamps = item.get("timestamp") or []
    indicators = item.get("indicators") or {}
    quote_list = indicators.get("quote") or [{}]
    quote_data = quote_list[0] if quote_list else {}
    adj_list = indicators.get("adjclose") or [{}]
    adj_closes = adj_list[0].get("adjclose") if adj_list else None

    opens = quote_data.get("open") or []
    highs = quote_data.get("high") or []
    lows = quote_data.get("low") or []
    closes = quote_data.get("close") or []
    volumes = quote_data.get("volume") or []

    bars_by_date: dict[str, OHLCVBar] = {}
    target_tz = TAIPEI_TZ if symbol.endswith(".TW") or symbol == "^TWII" else timezone.utc

    for idx, ts in enumerate(timestamps):
        if idx >= len(closes):
            break
        close_v = closes[idx]
        if close_v is None:
            continue
        open_v = opens[idx] if idx < len(opens) and opens[idx] is not None else close_v
        high_v = highs[idx] if idx < len(highs) and highs[idx] is not None else close_v
        low_v = lows[idx] if idx < len(lows) and lows[idx] is not None else close_v
        vol_v = volumes[idx] if idx < len(volumes) and volumes[idx] is not None else 0.0
        adj_v = (
            adj_closes[idx]
            if adj_closes is not None and idx < len(adj_closes) and adj_closes[idx] is not None
            else close_v
        )
        dt_str = datetime.fromtimestamp(int(ts), tz=target_tz).strftime("%Y-%m-%d")
        bars_by_date[dt_str] = OHLCVBar(
            date=dt_str,
            open=float(open_v),
            high=float(high_v),
            low=float(low_v),
            close=float(close_v),
            adj_close=float(adj_v),
            volume=float(vol_v),
        )

    sorted_bars = [bars_by_date[d] for d in sorted(bars_by_date)]
    # 若為台股商品且當前為週一至週五 13:45 收盤結算前，剔除當日尚未收盤的盤中殘缺 K 棒
    is_tw_symbol = symbol == "^TWII" or symbol.endswith(".TW")
    if is_tw_symbol and len(sorted_bars) > 1:
        ref_now = now_taipei.astimezone(TAIPEI_TZ) if now_taipei else datetime.now(TAIPEI_TZ)
        if ref_now.weekday() < 5 and (ref_now.hour, ref_now.minute) < (13, 45):
            today_str = ref_now.strftime("%Y-%m-%d")
            if sorted_bars[-1].date == today_str:
                sorted_bars.pop()

    if not sorted_bars:
        raise ValueError(f"Yahoo Finance 無法解析有效 K 線 ({symbol})")
    return MarketSeries(symbol=symbol, bars=sorted_bars)


def parse_twse_taiex_hist_rows(rows: list[list[Any]]) -> list[OHLCVBar]:
    """解析 TWSE MI_5MINS_HIST 資料列：[日期, 開盤指數, 最高指數, 最低指數, 收盤指數]。"""
    out: list[OHLCVBar] = []
    for row in rows:
        if len(row) < 5:
            continue
        dt_iso = roc_to_iso_date(str(row[0]))
        o = parse_number(row[1])
        h = parse_number(row[2])
        lo = parse_number(row[3])
        c = parse_number(row[4])
        if dt_iso and o is not None and h is not None and lo is not None and c is not None:
            out.append(OHLCVBar(date=dt_iso, open=o, high=h, low=lo, close=c, adj_close=c, volume=0.0))
    return out


def parse_twse_stock_day_rows(rows: list[list[Any]]) -> list[OHLCVBar]:
    """解析 TWSE STOCK_DAY 資料列：[日期, 成交股數, 成交金額, 開盤, 最高, 最低, 收盤, ...]。"""
    out: list[OHLCVBar] = []
    for row in rows:
        if len(row) < 7:
            continue
        dt_iso = roc_to_iso_date(str(row[0]))
        vol = parse_number(row[1]) or 0.0
        o = parse_number(row[3])
        h = parse_number(row[4])
        lo = parse_number(row[5])
        c = parse_number(row[6])
        if dt_iso and o is not None and h is not None and lo is not None and c is not None:
            out.append(OHLCVBar(date=dt_iso, open=o, high=h, low=lo, close=c, adj_close=c, volume=vol))
    return out


def _recent_month_strings(months: int = 15, ref_date: date | None = None) -> list[str]:
    base = ref_date or datetime.now(TAIPEI_TZ).date()
    result: list[str] = []
    y, m = base.year, base.month
    for _ in range(months):
        result.append(f"{y:04d}{m:02d}01")
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    result.reverse()
    return result


def _fetch_twse_taiex_fallback(client: httpx.Client, *, months: int = 15) -> MarketSeries:
    bars_by_date: dict[str, OHLCVBar] = {}
    for m_str in _recent_month_strings(months=months):
        resp = fetch_with_retry(
            client,
            TWSE_TAIEX_HIST_URL,
            params={"response": "json", "date": m_str},
            retries=1,
        )
        payload = resp.json()
        if payload.get("stat") == "OK":
            for b in parse_twse_taiex_hist_rows(payload.get("data") or []):
                bars_by_date[b.date] = b
        time.sleep(0.15)
    sorted_bars = [bars_by_date[d] for d in sorted(bars_by_date)]
    if len(sorted_bars) < 60:
        raise ValueError(f"TWSE MI_5MINS_HIST 備援 K 線筆數不足：{len(sorted_bars)}")
    return MarketSeries(symbol="^TWII", bars=sorted_bars)


def _fetch_twse_0050_fallback(client: httpx.Client, *, months: int = 15) -> MarketSeries:
    bars_by_date: dict[str, OHLCVBar] = {}
    for m_str in _recent_month_strings(months=months):
        resp = fetch_with_retry(
            client,
            TWSE_STOCK_DAY_URL,
            params={"response": "json", "date": m_str, "stockNo": "0050"},
            retries=1,
        )
        payload = resp.json()
        if payload.get("stat") == "OK":
            for b in parse_twse_stock_day_rows(payload.get("data") or []):
                bars_by_date[b.date] = b
        time.sleep(0.15)
    sorted_bars = [bars_by_date[d] for d in sorted(bars_by_date)]
    if len(sorted_bars) < 60:
        raise ValueError(f"TWSE STOCK_DAY 0050 備援 K 線筆數不足：{len(sorted_bars)}")
    return MarketSeries(symbol="0050.TW", bars=sorted_bars)


def fetch_single_yahoo_symbol(
    client: httpx.Client,
    symbol: str,
    *,
    range_str: str = "5y",
    interval: str = "1d",
) -> MarketSeries:
    """使用驗證通過之簡潔標頭向主／備援 Yahoo Finance 主機擷取單一商品日 K 資料。"""
    encoded = quote(symbol, safe="")
    last_err: Exception | None = None
    for idx, host in enumerate(YAHOO_HOSTS):
        url = f"https://{host}/v8/finance/chart/{encoded}"
        try:
            resp = client.get(
                url,
                params={"range": range_str, "interval": interval},
                headers=YAHOO_HEADERS,
            )
            resp.raise_for_status()
            return parse_yahoo_chart_payload(symbol, resp.json())
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if idx == 0:
                time.sleep(1.0)
    assert last_err is not None
    raise last_err


def compute_tsmc_adr_premium(
    tsm_adr_usd: float | None,
    tsmc_tw_twd: float | None,
    usd_twd: float | None,
) -> float | None:
    """計算台積電 ADR 換算台股溢價率（%）。1 單位 TSM ADR = 5 股 2330 普通股。"""
    if not tsm_adr_usd or not tsmc_tw_twd or not usd_twd:
        return None
    if tsmc_tw_twd <= 0:
        return None
    adr_implied_twd = (tsm_adr_usd * usd_twd) / 5.0
    return (adr_implied_twd / tsmc_tw_twd - 1.0) * 100.0


def collect_yahoo_market_data(
    *,
    range_str: str = "5y",
    timeout_sec: float = 12.0,
    throttle_sec: float = 0.35,
) -> tuple[dict[str, MarketSeries], dict[str, Any], CollectorReport]:
    """循序節流擷取核心市場、跨市場指標與 30 檔台股代表籃子，必要時啟用 TWSE 官方歷史備援。"""
    fetched_at = now_taipei_iso()
    core_list = list(dict.fromkeys(CORE_MARKET_SYMBOLS.values()))
    basket_only = [s for s in TAIWAN_BREADTH_BASKET if s not in core_list]
    ordered_symbols = core_list + basket_only

    series_map: dict[str, MarketSeries] = {}
    errors: dict[str, str] = {}
    domains_used = list(YAHOO_HOSTS)
    consecutive_basket_failures = 0

    with httpx.Client(timeout=timeout_sec, follow_redirects=True) as client:
        for idx, sym in enumerate(ordered_symbols):
            if idx > 0 and throttle_sec > 0:
                time.sleep(throttle_sec)
            sym_range = range_str if sym in {"^TWII", "0050.TW"} else "2y"
            try:
                series_map[sym] = fetch_single_yahoo_symbol(client, sym, range_str=sym_range)
                if sym in basket_only:
                    consecutive_basket_failures = 0
            except Exception as exc:  # noqa: BLE001
                errors[sym] = str(exc)
                if sym in basket_only:
                    consecutive_basket_failures += 1
                    if consecutive_basket_failures >= 4:
                        for rem in ordered_symbols[idx + 1 :]:
                            errors[rem] = "因連續限流保護而跳過剩餘代表籃子"
                        break

        # 若核心台股指數於 Yahoo 失敗，自動切換證交所官方歷史行情端點補位
        if "^TWII" not in series_map:
            try:
                series_map["^TWII"] = _fetch_twse_taiex_fallback(client, months=15)
                errors.pop("^TWII", None)
                if "www.twse.com.tw" not in domains_used:
                    domains_used.append("www.twse.com.tw")
            except Exception as fb_exc:  # noqa: BLE001
                errors["^TWII_fallback"] = str(fb_exc)

        if "0050.TW" not in series_map:
            try:
                series_map["0050.TW"] = _fetch_twse_0050_fallback(client, months=15)
                errors.pop("0050.TW", None)
                if "www.twse.com.tw" not in domains_used:
                    domains_used.append("www.twse.com.tw")
            except Exception as fb_exc:  # noqa: BLE001
                errors["0050.TW_fallback"] = str(fb_exc)

    tsm_latest = series_map["TSM"].bars[-1].close if "TSM" in series_map else None
    tsmc_tw_latest = series_map["2330.TW"].bars[-1].close if "2330.TW" in series_map else None
    usd_twd_latest = series_map["TWD=X"].bars[-1].close if "TWD=X" in series_map else None
    adr_premium_pct = compute_tsmc_adr_premium(tsm_latest, tsmc_tw_latest, usd_twd_latest)

    has_taiex = "^TWII" in series_map and len(series_map["^TWII"].bars) > 60
    has_0050 = "0050.TW" in series_map and len(series_map["0050.TW"].bars) > 60

    if not has_taiex or not has_0050:
        status = "failed"
    elif errors:
        status = "degraded"
    else:
        status = "ok"

    summary: dict[str, Any] = {
        "fetched_symbols": len(series_map),
        "failed_symbols": errors,
        "tsmc_adr_premium_pct": round(adr_premium_pct, 3) if adr_premium_pct is not None else None,
        "usd_twd_latest": round(usd_twd_latest, 4) if usd_twd_latest is not None else None,
    }
    msg = f"成功擷取 {len(series_map)}/{len(ordered_symbols)} 檔市場序列"
    if errors:
        msg += f"（失敗 {len(errors)} 檔）"

    report = CollectorReport(
        source_name="yahoo_finance",
        status=status,
        fetched_at=fetched_at,
        message=msg,
        domains_used=domains_used,
    )
    return series_map, summary, report
