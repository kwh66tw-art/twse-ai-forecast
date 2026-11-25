"""行為心理學與即時輿情／事件日曆擷取器：PTT 股板、Google News 台股 RSS 與固定市場行事曆。"""

from __future__ import annotations

import calendar
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from html import unescape
from typing import Any

import httpx

from twse_forecast.collectors.http_utils import fetch_with_retry, now_taipei_iso
from twse_forecast.config import PTT_EUPHORIA_KEYWORDS, PTT_PANIC_KEYWORDS, TAIPEI_TZ
from twse_forecast.models import CollectorReport

PTT_STOCK_INDEX = "https://www.ptt.cc/bbs/Stock/index.html"
GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search"
    "?q=台股+OR+台積電+OR+聯準會+OR+半導體+OR+關稅"
    "&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
)

_PREV_PAGE_RE = re.compile(r'href="(/bbs/Stock/index(\d+)\.html)"[^>]*>&lsaquo;\s*上頁')
_ENTRY_RE = re.compile(
    r'<div class="r-ent">(.*?)<div class="meta">',
    re.DOTALL,
)
_NREC_RE = re.compile(r'<div class="nrec">(?:<span[^>]*>(.*?)</span>)?</div>', re.DOTALL)
_TITLE_RE = re.compile(r'<div class="title">\s*(?:<a[^>]*>(.*?)</a>|(.*?))\s*</div>', re.DOTALL)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")


def _parse_nrec(raw: str) -> int:
    text = raw.strip()
    if not text:
        return 0
    if text == "爆":
        return 100
    if text == "XX":
        return -100
    if text.startswith("X"):
        try:
            return -10 * int(text[1:])
        except ValueError:
            return -10
    try:
        return int(text)
    except ValueError:
        return 0


def parse_ptt_stock_html(html_pages: list[str]) -> dict[str, Any]:
    """解析 PTT Stock 多頁 HTML，計算散戶極端情緒與反向指標。"""
    posts: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    for html_text in html_pages:
        for block in _ENTRY_RE.findall(html_text):
            title_m = _TITLE_RE.search(block)
            if not title_m:
                continue
            raw_title = title_m.group(1) or title_m.group(2) or ""
            clean_title = unescape(_TAG_STRIP_RE.sub("", raw_title)).strip()
            if not clean_title or "本文已被刪除" in clean_title or "[公告]" in clean_title:
                continue
            if "閒聊" in clean_title and "盤中" in clean_title:
                continue
            if clean_title in seen_titles:
                continue
            seen_titles.add(clean_title)

            nrec_m = _NREC_RE.search(block)
            nrec_val = _parse_nrec(nrec_m.group(1) if nrec_m and nrec_m.group(1) else "")
            posts.append({"title": clean_title, "nrec": nrec_val})

    if not posts:
        raise ValueError("PTT Stock 未解析出任何有效貼文")

    panic_hits = 0
    euphoria_hits = 0
    boom_posts = 0
    hot_panic_titles: list[str] = []
    hot_euphoria_titles: list[str] = []

    for p in posts:
        title_lower = p["title"].lower()
        is_boom = p["nrec"] >= 99
        if is_boom:
            boom_posts += 1
        if any(kw in title_lower for kw in PTT_PANIC_KEYWORDS):
            panic_hits += 1
            if len(hot_panic_titles) < 5:
                hot_panic_titles.append(p["title"])
        if any(kw in title_lower for kw in PTT_EUPHORIA_KEYWORDS):
            euphoria_hits += 1
            if len(hot_euphoria_titles) < 5:
                hot_euphoria_titles.append(p["title"])

    total = len(posts)
    avg_nrec = sum(p["nrec"] for p in posts) / total
    # 散戶狂熱度 (-100 極度恐慌 ~ +100 極度狂熱)
    raw_mood = ((euphoria_hits - panic_hits) / max(total, 10)) * 300.0
    retail_euphoria_score = max(-100.0, min(100.0, raw_mood))

    # 反向指標（極度恐慌 -> 正向短線修復訊號；極度狂熱 -> 負向過熱警戒訊號）
    contrarian_signal = 0.0
    if retail_euphoria_score <= -20.0:
        contrarian_signal = min(100.0, (-retail_euphoria_score - 15.0) * 1.4)
    elif retail_euphoria_score >= 25.0:
        contrarian_signal = -min(100.0, (retail_euphoria_score - 20.0) * 1.3)

    return {
        "posts_analyzed": total,
        "boom_posts": boom_posts,
        "avg_nrec": round(avg_nrec, 1),
        "panic_keyword_hits": panic_hits,
        "euphoria_keyword_hits": euphoria_hits,
        "retail_euphoria_score": round(retail_euphoria_score, 1),
        "contrarian_signal": round(contrarian_signal, 1),
        "sample_panic_titles": hot_panic_titles,
        "sample_euphoria_titles": hot_euphoria_titles,
    }


def parse_google_news_rss(xml_text: str, *, limit: int = 18) -> list[dict[str, str]]:
    """解析 Google News 台股與總經 RSS 頭條。"""
    root = ET.fromstring(xml_text)
    items: list[dict[str, str]] = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        source_el = item.find("source")
        source = (source_el.text or "").strip() if source_el is not None else ""
        if title:
            items.append({"title": title, "source": source, "pub_date": pub_date})
        if len(items) >= limit:
            break
    return items


def _third_wednesday(year: int, month: int) -> date:
    cal = calendar.monthcalendar(year, month)
    wednesdays = [week[calendar.WEDNESDAY] for week in cal if week[calendar.WEDNESDAY] != 0]
    return date(year, month, wednesdays[2])


def build_tw_event_calendar(ref_date: date | None = None, horizon_days: int = 28) -> list[dict[str, str]]:
    """計算未來 4 週台股固定結構性事件日曆。"""
    start = ref_date or datetime.now(TAIPEI_TZ).date()
    end = start + timedelta(days=horizon_days)
    events: list[dict[str, str]] = []

    # 1. 台指期貨與選擇權月結算日（每月第三個週三）
    for offset_m in (0, 1):
        y = start.year + (start.month - 1 + offset_m) // 12
        m = (start.month - 1 + offset_m) % 12 + 1
        settle_day = _third_wednesday(y, m)
        if start <= settle_day <= end:
            events.append({
                "date": settle_day.isoformat(),
                "event": "台指期／月選擇權結算日（常伴隨外資轉倉與尾盤波動放大）",
            })

    # 2. 檢查是否涵蓋每月 1~10 日上市櫃營收公布期
    d_cursor = start
    revenue_window_dates: list[date] = []
    while d_cursor <= end:
        if 1 <= d_cursor.day <= 10:
            revenue_window_dates.append(d_cursor)
        d_cursor += timedelta(days=1)
    if revenue_window_dates:
        events.append({
            "date": f"{revenue_window_dates[0].isoformat()} ~ {revenue_window_dates[-1].isoformat()}",
            "event": "上市櫃前月營收密集公布期（10 日截止，決定 1–3 個月動能強弱）",
        })

    # 3. 季報申報截止提示
    deadlines = [(3, 31, "年度財報"), (5, 15, "第一季財報"), (8, 14, "第二季財報"), (11, 14, "第三季財報")]
    for m, day, label in deadlines:
        for y in (start.year, start.year + 1):
            dl = date(y, m, day)
            if start <= dl <= end:
                events.append({
                    "date": dl.isoformat(),
                    "event": f"台股{label}法定公告截止日",
                })

    # 4. 聯準會 FOMC 例會月份提示
    fomc_months = {1, 3, 5, 6, 7, 9, 11, 12}
    if start.month in fomc_months or end.month in fomc_months:
        events.append({
            "date": start.strftime("%Y-%m"),
            "event": "美國 FOMC 利率決議例行月份（須留意點陣圖與美債殖利率波動）",
        })

    return events


def collect_sentiment_and_news(
    *,
    timeout_sec: float = 15.0,
    pages: int = 5,
    ref_date: date | None = None,
) -> tuple[dict[str, Any], CollectorReport]:
    """擷取 PTT 股板近期情緒、Google News 台灣財經頭條與未來 4 週事件日曆。"""
    fetched_at = now_taipei_iso()
    ptt_data: dict[str, Any] | None = None
    news_items: list[dict[str, str]] = []
    errors: dict[str, str] = {}

    with httpx.Client(timeout=timeout_sec, follow_redirects=True) as client:
        # 1. PTT Stock 最新 5 頁
        try:
            resp = fetch_with_retry(
                client,
                PTT_STOCK_INDEX,
                cookies={"over18": "1"},
                retries=1,
            )
            html_pages = [resp.text]
            prev_m = _PREV_PAGE_RE.search(resp.text)
            if prev_m:
                prev_num = int(prev_m.group(2))
                for p_idx in range(prev_num, max(1, prev_num - (pages - 1)), -1):
                    try:
                        p_resp = fetch_with_retry(
                            client,
                            f"https://www.ptt.cc/bbs/Stock/index{p_idx}.html",
                            cookies={"over18": "1"},
                            retries=0,
                        )
                        html_pages.append(p_resp.text)
                    except Exception:  # noqa: BLE001
                        continue
            ptt_data = parse_ptt_stock_html(html_pages)
        except Exception as exc:  # noqa: BLE001
            errors["ptt_stock"] = str(exc)

        # 2. Google News 台灣財經 RSS
        try:
            rss_resp = fetch_with_retry(client, GOOGLE_NEWS_RSS, retries=1)
            news_items = parse_google_news_rss(rss_resp.text, limit=18)
        except Exception as exc:  # noqa: BLE001
            errors["google_news"] = str(exc)

    calendar_events = build_tw_event_calendar(ref_date=ref_date)

    ok_parts = (1 if ptt_data is not None else 0) + (1 if news_items else 0)
    if ok_parts == 2:
        status = "ok"
    elif ok_parts == 1:
        status = "degraded"
    else:
        status = "failed"

    msg = f"情緒與新聞取得 {ok_parts}/2 項（PTT={ptt_data is not None}，新聞={len(news_items)}則）"
    if errors:
        msg += f"；缺漏：{', '.join(errors.keys())}"

    report = CollectorReport(
        source_name="sentiment_news",
        status=status,
        fetched_at=fetched_at,
        message=msg,
        domains_used=["www.ptt.cc", "news.google.com"],
    )
    return {
        "ptt_stock": ptt_data,
        "news_headlines": news_items,
        "event_calendar": calendar_events,
        "errors": errors,
    }, report
