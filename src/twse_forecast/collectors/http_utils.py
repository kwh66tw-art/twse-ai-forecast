"""HTTP 請求工具與台股數值解析共用函式。"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import httpx

from twse_forecast.config import TAIPEI_TZ

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
}


def now_taipei_iso() -> str:
    """回傳目前台灣時間之 ISO 8601 字串（含時區）。"""
    return datetime.now(TAIPEI_TZ).isoformat(timespec="seconds")


def parse_number(raw: Any) -> float | None:
    """將含千分位逗號、百分比符號或空值標記的字串安全解析為浮點數。"""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        return None if val != val else val
    text = str(raw).strip()
    if not text or text in {"--", "-", "N/A", "NaN", "null", "None", "－", "X"}:
        return None
    text = text.replace(",", "").replace("%", "").replace("+", "").strip()
    try:
        val = float(text)
        return None if val != val else val
    except ValueError:
        return None


def fetch_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    retries: int = 2,
    backoff_sec: float = 0.8,
) -> httpx.Response:
    """帶有限次重試與指數退避的 HTTP GET 請求。"""
    merged_headers = dict(DEFAULT_HEADERS)
    if headers:
        merged_headers.update(headers)

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.get(
                url,
                params=params,
                headers=merged_headers,
                cookies=cookies,
            )
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if attempt < retries:
                wait_sec = (
                    2.5 * (attempt + 1)
                    if exc.response.status_code == 429
                    else backoff_sec * (attempt + 1)
                )
                time.sleep(wait_sec)
        except (httpx.HTTPError, ValueError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff_sec * (attempt + 1))
    assert last_exc is not None
    raise last_exc
