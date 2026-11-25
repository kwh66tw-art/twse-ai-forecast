"""台灣證券交易所（TWSE）公開資訊擷取器：估值、三大法人、融資餘額、營收動能與漲跌家數。"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from statistics import median
from typing import Any

import httpx

from twse_forecast.collectors.http_utils import fetch_with_retry, now_taipei_iso, parse_number
from twse_forecast.config import TAIPEI_TZ
from twse_forecast.models import CollectorReport

TWSE_OPENAPI_BWIBBU = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"
TWSE_OPENAPI_REVENUE = "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"
TWSE_RWD_BFI82U = "https://www.twse.com.tw/rwd/zh/fund/BFI82U"
TWSE_RWD_MARGN = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"
TWSE_RWD_MI_INDEX = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"


def parse_bwibbu_all(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """解析全體上市個股本益比、殖利率與股價淨值比彙整統計。"""
    pes: list[float] = []
    pbs: list[float] = []
    yields: list[float] = []

    for row in rows:
        pe = parse_number(row.get("PEratio") or row.get("本益比"))
        pb = parse_number(row.get("PBratio") or row.get("股價淨值比"))
        dy = parse_number(row.get("DividendYield") or row.get("殖利率(%)"))
        if pe is not None and 0 < pe < 300:
            pes.append(pe)
        if pb is not None and 0 < pb < 50:
            pbs.append(pb)
        if dy is not None and 0 <= dy < 30:
            yields.append(dy)

    if not pbs and not pes:
        raise ValueError("BWIBBU_ALL 無有效估值資料")

    return {
        "sample_count": len(rows),
        "median_pe": round(median(pes), 2) if pes else None,
        "median_pb": round(median(pbs), 2) if pbs else None,
        "median_dividend_yield_pct": round(median(yields), 2) if yields else None,
        "mean_dividend_yield_pct": round(sum(yields) / len(yields), 2) if yields else None,
    }


def parse_bfi82u_payload(payload: dict[str, Any]) -> dict[str, float] | None:
    """解析 TWSE BFI82U 三大法人買賣金額統計（回傳單位：億元新台幣）。"""
    if payload.get("stat") != "OK":
        return None
    rows = payload.get("data") or []
    result: dict[str, float] = {
        "foreign_net_b": 0.0,
        "trust_net_b": 0.0,
        "dealer_prop_net_b": 0.0,
        "dealer_hedge_net_b": 0.0,
        "total_net_b": 0.0,
    }
    matched = False
    for row in rows:
        if not row or len(row) < 4:
            continue
        name = str(row[0]).strip()
        diff_twd = parse_number(row[3])
        if diff_twd is None:
            continue
        diff_b = diff_twd / 1e8  # 元換算為億元
        if "外資及陸資(不含外資自營商)" in name or ("外資及陸資" in name and "合計" not in name):
            result["foreign_net_b"] += diff_b
            matched = True
        elif "投信" in name:
            result["trust_net_b"] += diff_b
            matched = True
        elif "自營商(自行買賣)" in name:
            result["dealer_prop_net_b"] += diff_b
            matched = True
        elif "自營商(避險)" in name:
            result["dealer_hedge_net_b"] += diff_b
            matched = True
        elif "合計" in name:
            result["total_net_b"] = diff_b
            matched = True

    if not matched:
        return None
    return {k: round(v, 2) for k, v in result.items()}


def parse_mi_margn_payload(payload: dict[str, Any]) -> dict[str, float] | None:
    """解析 TWSE MI_MARGN 信用交易統計中之融資金額（換算為億元新台幣）。"""
    if payload.get("stat") != "OK":
        return None
    tables = payload.get("tables") or []
    # 亦相容舊格式 data 直接在頂層
    candidate_rows: list[list[Any]] = []
    for tbl in tables:
        for r in tbl.get("data") or []:
            candidate_rows.append(r)
    if not candidate_rows and isinstance(payload.get("data"), list):
        candidate_rows = payload["data"]

    for row in candidate_rows:
        if not row or len(row) < 6:
            continue
        label = str(row[0]).strip()
        if "融資金額" in label:
            # 欄位順序通常為：[項目, 買進, 賣出, 現金償還, 前日餘額, 今日餘額]（單位：仟元）
            prev_k = parse_number(row[4])
            curr_k = parse_number(row[5])
            if curr_k is None:
                continue
            curr_b = curr_k / 1e5  # 仟元 -> 億元
            prev_b = (prev_k / 1e5) if prev_k is not None else curr_b
            chg_b = curr_b - prev_b
            chg_pct = (chg_b / prev_b * 100.0) if prev_b > 0 else 0.0
            return {
                "margin_balance_b": round(curr_b, 2),
                "margin_change_1d_b": round(chg_b, 2),
                "margin_change_1d_pct": round(chg_pct, 3),
            }
    return None


def parse_monthly_revenue_opendata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """解析上市公司每月營收彙總 OpenData，計算營收廣度與整體年增率。"""
    yoy_list: list[float] = []
    pos_count = 0
    total_curr = 0.0
    total_prev = 0.0
    latest_ym = ""

    for row in rows:
        ym = str(row.get("資料年月") or row.get("YearMonth") or "").strip()
        if ym > latest_ym:
            latest_ym = ym
        yoy = parse_number(
            row.get("營業收入-去年同月增減(%)")
            or row.get("去年同月增減(%)")
            or row.get("YoY")
        )
        curr_rev = parse_number(row.get("營業收入-當月營收") or row.get("當月營收"))
        prev_rev = parse_number(row.get("營業收入-去年當月營收") or row.get("去年當月營收"))

        if yoy is not None and -99.9 <= yoy <= 5000.0:
            yoy_list.append(yoy)
            if yoy > 0:
                pos_count += 1
        if curr_rev is not None and prev_rev is not None and curr_rev >= 0 and prev_rev > 0:
            total_curr += curr_rev
            total_prev += prev_rev

    if not yoy_list:
        raise ValueError("月營收 OpenData 無有效 YoY 欄位")

    agg_yoy = ((total_curr / total_prev) - 1.0) * 100.0 if total_prev > 0 else None
    return {
        "data_year_month": latest_ym or "unknown",
        "company_count": len(yoy_list),
        "positive_yoy_ratio_pct": round(pos_count / len(yoy_list) * 100.0, 2),
        "median_yoy_pct": round(median(yoy_list), 2),
        "aggregate_yoy_pct": round(agg_yoy, 2) if agg_yoy is not None else None,
    }


def parse_mi_index_breadth(payload: dict[str, Any]) -> dict[str, int] | None:
    """從 MI_INDEX 解析大盤上漲／下跌／平盤家數。"""
    if payload.get("stat") != "OK":
        return None
    for tbl in payload.get("tables") or []:
        title = str(tbl.get("title") or "")
        if "漲跌證券數合計" not in title:
            continue
        up_cnt = down_cnt = flat_cnt = 0
        for row in tbl.get("data") or []:
            if not row or len(row) < 3:
                continue
            item_name = str(row[0])
            # 解析「整體市場」列（格式常如 "620(18)"，括弧內為漲停）
            val_text = str(row[2]).split("(")[0].strip()
            num = parse_number(val_text)
            if num is None:
                continue
            if "上漲" in item_name:
                up_cnt = int(num)
            elif "下跌" in item_name:
                down_cnt = int(num)
            elif "持平" in item_name:
                flat_cnt = int(num)
        if up_cnt + down_cnt > 0:
            return {"advancing": up_cnt, "declining": down_cnt, "unchanged": flat_cnt}
    return None


def _recent_candidate_dates(today: date | None = None, lookback_days: int = 12) -> list[str]:
    base = today or datetime.now(TAIPEI_TZ).date()
    dates: list[str] = []
    for offset in range(lookback_days):
        d = base - timedelta(days=offset)
        if d.weekday() < 5:  # 跳過週六、週日
            dates.append(d.strftime("%Y%m%d"))
    return dates


def collect_twse_official_data(
    *,
    timeout_sec: float = 15.0,
    today: date | None = None,
) -> tuple[dict[str, Any], CollectorReport]:
    """擷取台灣證交所估值、三大法人近 5 日流向、融資餘額、營收廣度與漲跌家數。"""
    fetched_at = now_taipei_iso()
    data: dict[str, Any] = {
        "valuation": None,
        "institutional_flow": None,
        "margin": None,
        "monthly_revenue": None,
        "market_breadth_counts": None,
    }
    errors: dict[str, str] = {}
    cand_dates = _recent_candidate_dates(today=today, lookback_days=14)

    with httpx.Client(timeout=timeout_sec, follow_redirects=True) as client:
        # 1. 大盤個股估值彙整 (BWIBBU_ALL)
        try:
            resp = fetch_with_retry(client, TWSE_OPENAPI_BWIBBU, retries=1)
            data["valuation"] = parse_bwibbu_all(resp.json())
        except Exception as exc:  # noqa: BLE001
            errors["valuation"] = str(exc)

        # 2. 上市公司月營收 OpenData
        try:
            resp = fetch_with_retry(client, TWSE_OPENAPI_REVENUE, retries=1)
            data["monthly_revenue"] = parse_monthly_revenue_opendata(resp.json())
        except Exception as exc:  # noqa: BLE001
            errors["monthly_revenue"] = str(exc)

        # 3. 三大法人近 5 個交易日淨買賣超 (BFI82U)
        try:
            daily_flows: list[dict[str, Any]] = []
            for d_str in cand_dates:
                if len(daily_flows) >= 5:
                    break
                resp = fetch_with_retry(
                    client,
                    TWSE_RWD_BFI82U,
                    params={"response": "json", "dayDate": d_str, "type": "day"},
                    retries=0,
                )
                parsed = parse_bfi82u_payload(resp.json())
                if parsed is not None:
                    daily_flows.append({"date": d_str, **parsed})
            if daily_flows:
                data["institutional_flow"] = {
                    "latest_date": daily_flows[0]["date"],
                    "days_collected": len(daily_flows),
                    "latest_1d": daily_flows[0],
                    "sum_5d_foreign_net_b": round(sum(x["foreign_net_b"] for x in daily_flows), 2),
                    "sum_5d_trust_net_b": round(sum(x["trust_net_b"] for x in daily_flows), 2),
                    "sum_5d_dealer_prop_net_b": round(
                        sum(x["dealer_prop_net_b"] for x in daily_flows), 2
                    ),
                    "sum_5d_total_net_b": round(sum(x["total_net_b"] for x in daily_flows), 2),
                }
            else:
                errors["institutional_flow"] = "近期無可用三大法人資料"
        except Exception as exc:  # noqa: BLE001
            errors["institutional_flow"] = str(exc)

        # 4. 融資餘額 (MI_MARGN) & 漲跌家數 (MI_INDEX)
        for d_str in cand_dates[:6]:
            if data["margin"] is None:
                try:
                    resp = fetch_with_retry(
                        client,
                        TWSE_RWD_MARGN,
                        params={"response": "json", "selectType": "MS", "date": d_str},
                        retries=0,
                    )
                    parsed_m = parse_mi_margn_payload(resp.json())
                    if parsed_m is not None:
                        data["margin"] = {"date": d_str, **parsed_m}
                except Exception as exc:  # noqa: BLE001
                    errors.setdefault("margin", str(exc))

            if data["market_breadth_counts"] is None:
                try:
                    resp = fetch_with_retry(
                        client,
                        TWSE_RWD_MI_INDEX,
                        params={"response": "json", "type": "ALLBUT0999", "date": d_str},
                        retries=0,
                    )
                    parsed_b = parse_mi_index_breadth(resp.json())
                    if parsed_b is not None:
                        data["market_breadth_counts"] = {"date": d_str, **parsed_b}
                except Exception as exc:  # noqa: BLE001
                    errors.setdefault("market_breadth_counts", str(exc))

            if data["margin"] is not None and data["market_breadth_counts"] is not None:
                break

    ok_count = sum(1 for v in data.values() if v is not None)
    if ok_count >= 4:
        status = "ok"
    elif ok_count >= 1:
        status = "degraded"
    else:
        status = "failed"

    msg = f"證交所資料取得 {ok_count}/5 項模組"
    if errors:
        msg += f"（部分缺漏：{', '.join(sorted(errors.keys()))}）"

    report = CollectorReport(
        source_name="twse_official",
        status=status,
        fetched_at=fetched_at,
        message=msg,
        domains_used=["openapi.twse.com.tw", "www.twse.com.tw"],
    )
    return {"data": data, "errors": errors}, report
