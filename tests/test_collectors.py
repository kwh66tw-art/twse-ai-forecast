"""資料擷取器與解析器單元測試。"""

from __future__ import annotations

from datetime import date

import pytest

from twse_forecast.collectors.macro_fred import parse_fred_csv
from twse_forecast.collectors.market_yahoo import (
    compute_tsmc_adr_premium,
    parse_twse_stock_day_rows,
    parse_twse_taiex_hist_rows,
    parse_yahoo_chart_payload,
    roc_to_iso_date,
)
from twse_forecast.collectors.sentiment_news import (
    build_tw_event_calendar,
    parse_google_news_rss,
    parse_ptt_stock_html,
)
from twse_forecast.collectors.taifex_official import (
    parse_taifex_fut_contracts_html,
    parse_taifex_inst_futures,
    parse_taifex_pc_ratio,
)
from twse_forecast.collectors.twse_official import (
    parse_bfi82u_payload,
    parse_bwibbu_all,
    parse_mi_index_breadth,
    parse_mi_margn_payload,
    parse_monthly_revenue_opendata,
)


def test_tsmc_adr_premium_calculation() -> None:
    # 1 TSM ADR = 5 股 2330.TW；若 ADR=180 USD, 匯率=32, 2330=1000 -> (180*32/5)/1000 - 1 = 15.2%
    prem = compute_tsmc_adr_premium(180.0, 1000.0, 32.0)
    assert prem is not None
    assert prem == pytest.approx(15.2, rel=1e-4)


def test_parse_yahoo_chart_payload() -> None:
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1700006400, 1700092800],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 102.0],
                                "high": [103.0, 105.0],
                                "low": [99.0, 101.0],
                                "close": [102.0, 104.0],
                                "volume": [1000, 1200],
                            }
                        ],
                        "adjclose": [{"adjclose": [100.5, 102.5]}],
                    },
                }
            ]
        }
    }
    series = parse_yahoo_chart_payload("^TWII", payload)
    assert series.symbol == "^TWII"
    assert len(series.bars) == 2
    assert series.bars[-1].adj_close == 102.5


def test_roc_date_and_twse_hist_fallback_parsers() -> None:
    assert roc_to_iso_date("115/11/24") == "2026-11-24"
    assert roc_to_iso_date("bad") is None

    taiex_bars = parse_twse_taiex_hist_rows([
        ["115/11/24", "22,500.00", "22,700.00", "22,400.00", "22,650.00"],
    ])
    assert len(taiex_bars) == 1
    assert taiex_bars[0].date == "2026-11-24"
    assert taiex_bars[0].close == 22650.0

    etf_bars = parse_twse_stock_day_rows([
        ["115/11/24", "15,000,000", "3,000,000,000", "194.00", "196.00", "193.50", "195.50", "+1.50", "12,000"],
    ])
    assert len(etf_bars) == 1
    assert etf_bars[0].close == 195.5
    assert etf_bars[0].volume == 15_000_000.0


def test_twse_parsers() -> None:
    val = parse_bwibbu_all([
        {"PEratio": "15.5", "PBratio": "1.6", "DividendYield": "4.2"},
        {"PEratio": "20.0", "PBratio": "2.0", "DividendYield": "3.0"},
    ])
    assert val["median_pb"] == 1.8
    assert val["median_pe"] == 17.75

    bfi = parse_bfi82u_payload({
        "stat": "OK",
        "data": [
            ["外資及陸資(不含外資自營商)", "100", "50", "15,000,000,000"],
            ["投信", "20", "10", "3,000,000,000"],
            ["自營商(自行買賣)", "10", "15", "-500,000,000"],
            ["自營商(避險)", "5", "5", "0"],
            ["合計", "135", "80", "17,500,000,000"],
        ],
    })
    assert bfi is not None
    assert bfi["foreign_net_b"] == 150.0
    assert bfi["trust_net_b"] == 30.0
    assert bfi["dealer_prop_net_b"] == -5.0

    margn = parse_mi_margn_payload({
        "stat": "OK",
        "tables": [
            {"data": [["融資金額(仟元)", "10", "5", "1", "30,000,000", "30,300,000"]]}
        ],
    })
    assert margn is not None
    assert margn["margin_balance_b"] == 303.0
    assert margn["margin_change_1d_b"] == 3.0

    rev = parse_monthly_revenue_opendata([
        {"資料年月": "11510", "營業收入-去年同月增減(%)": "12.5", "營業收入-當月營收": "110", "營業收入-去年當月營收": "100"},
        {"資料年月": "11510", "營業收入-去年同月增減(%)": "-5.0", "營業收入-當月營收": "95", "營業收入-去年當月營收": "100"},
    ])
    assert rev["positive_yoy_ratio_pct"] == 50.0
    assert rev["aggregate_yoy_pct"] == 2.5

    breadth = parse_mi_index_breadth({
        "stat": "OK",
        "tables": [
            {
                "title": "漲跌證券數合計",
                "data": [
                    ["上漲(漲停)", "100(5)", "620(18)"],
                    ["下跌(跌停)", "50(1)", "310(4)"],
                ],
            }
        ],
    })
    assert breadth == {"advancing": 620, "declining": 310, "unchanged": 0}


def test_taifex_and_fred_parsers() -> None:
    pc = parse_taifex_pc_ratio([
        {"Date": "20261120", "PutCallOIRatio%": "95.5"},
        {"Date": "20261123", "PutCallOIRatio%": "112.4"},
    ])
    assert pc["latest_pc_ratio_pct"] == 112.4

    fut = parse_taifex_inst_futures([
        {"日期": "20261123", "契約名稱": "臺股期貨", "身份別": "外資", "多空淨額未平倉口數": "-32,450"},
    ])
    assert fut["foreign_tx_net_oi"] == -32450

    fut_html = """
    <div>查詢日期 2026/11/23</div>
    <table>
      <tr><td rowspan="3">1</td><td rowspan="3">臺股期貨</td><td>自營商</td>
          <td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td>
          <td>7</td><td>8</td><td>9</td><td>10</td><td>-233</td><td>-1000</td></tr>
      <tr><td>投信</td>
          <td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td>
          <td>7</td><td>8</td><td>9</td><td>10</td><td>98,014</td><td>2000</td></tr>
      <tr><td>外資</td>
          <td>68,962</td><td>539</td><td>69,654</td><td>545</td><td>-692</td><td>-5</td>
          <td>12,852</td><td>103</td><td>116,034</td><td>935</td><td>-103,182</td><td>-832,157</td></tr>
    </table>
    """
    fut_parsed = parse_taifex_fut_contracts_html(fut_html)
    assert fut_parsed["latest_date"] == "20261123"
    assert fut_parsed["foreign_tx_net_oi"] == -103182

    fred = parse_fred_csv("DATE,DGS10\n2026-11-20,4.25\n2026-11-21,.\n2026-11-22,4.30\n")
    assert fred == [("2026-11-20", 4.25), ("2026-11-22", 4.30)]


def test_ptt_sentiment_rss_and_calendar() -> None:
    sample_html = """
    <div class="r-ent">
      <div class="nrec"><span class="hl f1">爆</span></div>
      <div class="title"><a href="/bbs/Stock/1.html">[心得] 融資斷頭畢業睡公園</a></div>
      <div class="meta"></div>
    </div>
    <div class="r-ent">
      <div class="nrec"><span class="hl f3">88</span></div>
      <div class="title"><a href="/bbs/Stock/2.html">[請益] 腳麻丸子大崩盤怎麼辦</a></div>
      <div class="meta"></div>
    </div>
    <div class="r-ent">
      <div class="nrec"><span class="hl f2">10</span></div>
      <div class="title"><a href="/bbs/Stock/3.html">[公告] 板規宣導</a></div>
      <div class="meta"></div>
    </div>
    """
    ptt = parse_ptt_stock_html([sample_html])
    assert ptt["posts_analyzed"] == 2  # 排除公告
    assert ptt["panic_keyword_hits"] == 2
    assert ptt["retail_euphoria_score"] < -20
    assert ptt["contrarian_signal"] > 0  # 極度恐慌產生正向反向訊號

    rss = """<?xml version="1.0" encoding="UTF-8"?>
    <rss><channel><item>
      <title>台積電法說釋出樂觀展望</title>
      <pubDate>Sun, 22 Nov 2026 08:00:00 GMT</pubDate>
      <source>工商時報</source>
    </item></channel></rss>"""
    news = parse_google_news_rss(rss)
    assert len(news) == 1
    assert news[0]["source"] == "工商時報"

    events = build_tw_event_calendar(ref_date=date(2026, 11, 2), horizon_days=28)
    assert any("營收" in e["event"] for e in events)
    assert any("結算日" in e["event"] for e in events)
