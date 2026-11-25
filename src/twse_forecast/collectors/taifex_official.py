"""台灣期貨交易所（TAIFEX）公開資訊擷取器：選擇權 Put/Call Ratio 與外資期貨淨未平倉。"""

from __future__ import annotations

import re
from html import unescape
from typing import Any

import httpx

from twse_forecast.collectors.http_utils import fetch_with_retry, now_taipei_iso, parse_number
from twse_forecast.models import CollectorReport

TAIFEX_PC_RATIO_API = "https://openapi.taifex.com.tw/v1/PutCallRatio"
TAIFEX_INST_FUT_API = "https://openapi.taifex.com.tw/v1/MarketDataOfInstitutionalInvestorsFuturesByDate"
TAIFEX_FUT_HTML_URL = "https://www.taifex.com.tw/cht/3/futContractsDate"

_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_DATE_RE = re.compile(r"(20\d{2})[/\-](\d{2})[/\-](\d{2})")


def parse_taifex_pc_ratio(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """解析期交所 PutCallRatio OpenAPI 回傳清單。"""
    valid: list[tuple[str, float]] = []
    for row in rows:
        d = str(row.get("Date") or row.get("日期") or "").strip()
        ratio = parse_number(
            row.get("PutCallOIRatio%")
            or row.get("PutCallRatio")
            or row.get("買賣權未平倉量比率%")
            or row.get("Put/CallRatio(%)")
            or row.get("PutCallVolumeRatio%")
        )
        if d and ratio is not None:
            valid.append((d, ratio))

    if not valid:
        raise ValueError("TAIFEX PutCallRatio 無有效資料")

    valid.sort(key=lambda x: x[0])
    recent_5 = [x[1] for x in valid[-5:]]
    return {
        "latest_date": valid[-1][0],
        "latest_pc_ratio_pct": round(valid[-1][1], 2),
        "avg_5d_pc_ratio_pct": round(sum(recent_5) / len(recent_5), 2),
    }


def parse_taifex_inst_futures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """解析期交所三大法人臺股期貨（TX）外資多空未平倉淨口數（JSON 格式）。"""
    tx_foreign_rows: list[tuple[str, float]] = []
    for row in rows:
        contract = str(
            row.get("ContractName")
            or row.get("ContractCode")
            or row.get("商品名稱")
            or row.get("契約名稱")
            or ""
        )
        identity = str(
            row.get("Identity")
            or row.get("TraderType")
            or row.get("身份別")
            or ""
        )
        if ("臺股期貨" in contract or "台股期貨" in contract or contract.strip() == "TX") and (
            "外資" in identity or "Foreign" in identity
        ):
            d = str(row.get("Date") or row.get("日期") or "").strip()
            net_oi = parse_number(
                row.get("NetOpenInterestContracts")
                or row.get("NetOI")
                or row.get("多空淨額未平倉口數")
                or row.get("未平倉餘額多空淨額口數")
            )
            if d and net_oi is not None:
                tx_foreign_rows.append((d, net_oi))

    if not tx_foreign_rows:
        raise ValueError("TAIFEX 外資臺股期貨未平倉欄位無法匹配或無資料")

    tx_foreign_rows.sort(key=lambda x: x[0])
    return {
        "latest_date": tx_foreign_rows[-1][0],
        "foreign_tx_net_oi": int(round(tx_foreign_rows[-1][1])),
    }


def parse_taifex_fut_contracts_html(html_text: str) -> dict[str, Any]:
    """解析期交所 futContractsDate HTML 表格，提取外資臺股期貨（TX）未平倉多空淨口數。"""
    trs = _TR_RE.findall(html_text)
    in_tx_block = False
    scanned_after_tx = 0

    date_match = _DATE_RE.search(html_text)
    latest_date = (
        f"{date_match.group(1)}{date_match.group(2)}{date_match.group(3)}"
        if date_match
        else "latest"
    )

    for tr_inner in trs:
        cells_raw = _TD_RE.findall(tr_inner)
        cells = [
            unescape(_TAG_RE.sub("", c)).strip()
            for c in cells_raw
        ]
        row_text = " ".join(cells)

        if "臺股期貨" in row_text or "台股期貨" in row_text:
            in_tx_block = True
            scanned_after_tx = 0
        elif in_tx_block:
            scanned_after_tx += 1
            if scanned_after_tx > 4:
                in_tx_block = False
                continue

        if in_tx_block and ("外資" in row_text):
            numeric_vals = [parse_number(c) for c in cells]
            nums = [v for v in numeric_vals if v is not None]
            # 外資列末 12 個數值最後兩欄依序為：[未平倉多空淨額口數, 未平倉多空淨額契約金額]
            if len(nums) >= 12:
                return {
                    "latest_date": latest_date,
                    "foreign_tx_net_oi": int(round(nums[-2])),
                    "source": "html_futContractsDate",
                }

    raise ValueError("無法於 TAIFEX futContractsDate HTML 中找到臺股期貨外資未平倉列")


def collect_taifex_data(*, timeout_sec: float = 15.0) -> tuple[dict[str, Any], CollectorReport]:
    """擷取台指選擇權 Put/Call Ratio 與外資臺股期貨淨未平倉口數。"""
    fetched_at = now_taipei_iso()
    data: dict[str, Any] = {"put_call_ratio": None, "foreign_futures": None}
    errors: dict[str, str] = {}

    with httpx.Client(timeout=timeout_sec, follow_redirects=True) as client:
        try:
            resp = fetch_with_retry(client, TAIFEX_PC_RATIO_API, retries=1)
            data["put_call_ratio"] = parse_taifex_pc_ratio(resp.json())
        except Exception as exc:  # noqa: BLE001
            errors["put_call_ratio"] = str(exc)

        # 優先透過正式網頁查詢表格解析外資臺股期貨未平倉；失敗再嘗試 OpenAPI JSON
        try:
            html_resp = fetch_with_retry(client, TAIFEX_FUT_HTML_URL, retries=1)
            data["foreign_futures"] = parse_taifex_fut_contracts_html(html_resp.text)
        except Exception as html_exc:  # noqa: BLE001
            try:
                api_resp = fetch_with_retry(client, TAIFEX_INST_FUT_API, retries=0)
                data["foreign_futures"] = parse_taifex_inst_futures(api_resp.json())
            except Exception as api_exc:  # noqa: BLE001
                errors["foreign_futures"] = f"HTML 失敗({html_exc}); OpenAPI 失敗({api_exc})"

    ok_count = sum(1 for v in data.values() if v is not None)
    if ok_count == 2:
        status = "ok"
    elif ok_count == 1:
        status = "degraded"
    else:
        status = "failed"

    msg = f"期交所資料取得 {ok_count}/2 項"
    if errors:
        msg += f"（缺漏：{', '.join(errors.keys())}）"

    report = CollectorReport(
        source_name="taifex_official",
        status=status,
        fetched_at=fetched_at,
        message=msg,
        domains_used=["openapi.taifex.com.tw", "www.taifex.com.tw"],
    )
    return {"data": data, "errors": errors}, report
