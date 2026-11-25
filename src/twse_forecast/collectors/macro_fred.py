"""聖路易聯準銀行（FRED）總經公開 CSV 擷取器（無需 API 金鑰）。"""

from __future__ import annotations

import csv
import io
from datetime import date, timedelta
from typing import Any

import httpx

from twse_forecast.collectors.http_utils import fetch_with_retry, now_taipei_iso, parse_number
from twse_forecast.models import CollectorReport

FRED_SERIES: dict[str, str] = {
    "dgs10": "DGS10",  # 美債 10 年期殖利率 (%)
    "t10y2y": "T10Y2Y",  # 美債 10Y-2Y 殖利率利差 (%)
    "hy_oas": "BAMLH0A0HYM2",  # 美國非投資等級債信用利差 (%)
}


def parse_fred_csv(csv_text: str) -> list[tuple[str, float]]:
    """解析 FRED fredgraph CSV 並過濾缺失值 '.'。"""
    reader = csv.reader(io.StringIO(csv_text))
    rows: list[tuple[str, float]] = []
    header_skipped = False
    for row in reader:
        if not row or len(row) < 2:
            continue
        if not header_skipped:
            header_skipped = True
            continue
        date_str = row[0].strip()
        val = parse_number(row[1])
        if val is not None:
            rows.append((date_str, val))
    return rows


def collect_fred_macro(*, timeout_sec: float = 20.0) -> tuple[dict[str, Any], CollectorReport]:
    """擷取美債 10 年殖利率、10Y-2Y 利差與高收益債信用利差。"""
    fetched_at = now_taipei_iso()
    metrics: dict[str, Any] = {}
    errors: dict[str, str] = {}
    clean_headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

    start_date = (date.today() - timedelta(days=730)).isoformat()
    with httpx.Client(timeout=timeout_sec, follow_redirects=True, headers=clean_headers) as client:
        for key, series_id in FRED_SERIES.items():
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start_date}"
            try:
                resp = client.get(url)
                resp.raise_for_status()
                obs = parse_fred_csv(resp.text)
                if not obs:
                    raise ValueError("無有效觀測值")
                latest_date, latest_val = obs[-1]
                prev_20 = obs[-21][1] if len(obs) > 20 else obs[0][1]
                prev_60 = obs[-61][1] if len(obs) > 60 else obs[0][1]
                metrics[key] = {
                    "series_id": series_id,
                    "date": latest_date,
                    "latest": round(latest_val, 4),
                    "change_20obs": round(latest_val - prev_20, 4),
                    "change_60obs": round(latest_val - prev_60, 4),
                }
            except Exception as exc:  # noqa: BLE001
                errors[key] = str(exc)

    if len(metrics) == len(FRED_SERIES):
        status = "ok"
    elif metrics:
        status = "degraded"
    else:
        status = "failed"

    msg = f"FRED 總經指標取得 {len(metrics)}/{len(FRED_SERIES)} 項"
    if errors:
        msg += f"（未取得：{', '.join(errors.keys())}）"

    report = CollectorReport(
        source_name="macro_fred",
        status=status,
        fetched_at=fetched_at,
        message=msg,
        domains_used=["fred.stlouisfed.org"],
    )
    return {"series": metrics, "errors": errors}, report
