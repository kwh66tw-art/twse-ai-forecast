"""每週預測報告（Markdown）產生器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from twse_forecast.config import HORIZONS


def render_weekly_report(
    *,
    record: dict[str, Any],
    features: dict[str, Any],
    audit_summary: dict[str, Any] | None = None,
) -> str:
    """產出結構清晰、台灣中文撰寫之每週大盤多週期預測週報。"""
    rec_id = record["record_id"]
    ai_f = record["ai_forecast"]
    q_b = record["quant_baseline"]
    cross = features.get("cross_market") or {}
    fund = features.get("fundamentals_valuation") or {}
    micro = features.get("microstructure_flows") or {}
    tech = features.get("technicals") or {}
    breadth = features.get("breadth") or {}
    sentiment = features.get("sentiment") or {}

    catchup_label = "非盤前補跑／基期建立" if record["is_late_catchup"] else "週一 08:00 盤前準時鎖定"

    lines: list[str] = [
        f"# 台股大盤多週期預測週報（{rec_id}）",
        "",
        "## 一、鎖定中繼資料與防竄改存證",
        f"- **週次識別碼**：`{rec_id}`（排程槽位 `{record['scheduled_slot']}`）",
        f"- **實際寫入時間**：`{record['executed_at_iso']}`（狀態：**{catchup_label}**）",
        f"- **基準收盤資訊**：`{record['reference_trade_date']}` 加權指數 `{record['reference_taiex_close']:,.2f}` 點 ／ 0050 還原收盤 `{record['reference_0050_adj_close']:,.4f}`",
        f"- **執行模型**：`{ai_f['model_name']}`",
        f"- **前筆雜湊 (`prev_hash`)**：`{record['prev_hash'][:16]}...`",
        f"- **本週鎖定雜湊 (`record_hash`)**：`{record['record_hash']}`",
        "",
        "## 二、五週期預測承諾表（AI 辯證 vs. 純量化基準）",
        f"- **建議模擬持股水位（0050.TW）**：AI 模型 **`{ai_f['target_allocation']*100:.1f}%`** ｜ 純量化基準 `{q_b['target_allocation']*100:.1f}%`",
        "",
        "| 週期 | 驗算錨點 | 量化分數 | AI 分數 (Δ) | 預期報酬 | 多／盤／空機率 | 80% 信賴點位區間 | 失效關鍵點位 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for spec in HORIZONS:
        qf = q_b["horizons"][spec.key]
        af = ai_f["horizons"][spec.key]
        lines.append(
            f"| **{spec.label}** | {spec.anchor_trading_days} 日 | `{qf['quant_score']:+.1f}` | "
            f"`{af['ai_score']:+.1f}` (`{af['delta_score']:+.1f}`) | "
            f"`{af['expected_return']*100:+.2f}%` | "
            f"`{af['prob_up']*100:.1f}% / {af['prob_flat']*100:.1f}% / {af['prob_down']*100:.1f}%` | "
            f"`{af['ci80_low_level']:,.0f} ~ {af['ci80_high_level']:,.0f}` | `{af['invalidation_level']:,.0f}` |"
        )

    lines.extend([
        "",
        "## 三、AI 多空辯證與投資長總結",
        f"### 多頭論點（Bull Case）\n{ai_f['bull_case']}",
        "",
        f"### 空頭稽核（Bear Case）\n{ai_f['bear_case']}",
        "",
        f"### 投資長綜合裁決（CIO Synthesis）\n{ai_f['cio_synthesis']}",
        "",
        f"### 與純量化基準差異說明（Override Rationale）\n{ai_f['override_rationale']}",
        "",
        "### 各週期核心論述",
    ])
    for spec in HORIZONS:
        af = ai_f["horizons"][spec.key]
        lines.append(f"- **{spec.label}**：{af['thesis']}")

    if ai_f.get("external_news_notes"):
        lines.extend(["", "### 本週額外檢索之突發消息／事件註記"])
        for note in ai_f["external_news_notes"]:
            lines.append(f"- {note}")

    lines.extend([
        "",
        "## 四、關鍵市場數據快照",
        f"- **跨市場連動**：台積電 ADR 溢價 `{cross.get('tsmc_adr_premium_pct')}%` ｜ 費半 5 日 `{cross.get('sox_ret_5d_pct')}%` ｜ 美元兌台幣 `{cross.get('usd_twd_latest')}` ｜ 美債 10Y `{cross.get('us10y_yield_pct')}%` ｜ VIX `{cross.get('vix_latest')}`",
        f"- **基本面估值**：上市中位數 P/B `{fund.get('market_median_pb')}` ｜ P/E `{fund.get('market_median_pe')}` ｜ 台股 ERP `{fund.get('equity_risk_premium_pct')}%` ｜ 上市月營收正成長廣度 `{fund.get('monthly_revenue_pos_ratio_pct')}%`",
        f"- **法人與籌碼**：近 5 日外資現貨 `{micro.get('foreign_5d_net_b')}` 億 ｜ 投信 `{micro.get('trust_5d_net_b')}` 億 ｜ 外資期貨淨未平倉 `{micro.get('foreign_tx_net_oi')}` 口 ｜ P/C Ratio `{micro.get('pc_ratio_latest_pct')}%`",
        f"- **技術與寬度**：20 日乖離 `{tech.get('bias_20d_pct')}%` ｜ 3–12 月動能 `{tech.get('momentum_12_1m_pct')}%` ｜ 季線扣抵 `{(tech.get('ma60_deduction') or {}).get('regime')}` ｜ 30 檔籃子站上季線 `{breadth.get('pct_above_ma60')}%`（拉積警示：`{breadth.get('narrow_leadership_warning')}`）",
    ])
    if sentiment:
        lines.append(
            f"- **PTT 股板心理學**：狂熱分數 `{sentiment.get('retail_euphoria_score')}` ｜ 反向訊號 `{sentiment.get('contrarian_signal')}`（爆文 `{sentiment.get('boom_posts')}` 篇）"
        )

    if audit_summary and audit_summary.get("total_settled_horizons", 0) > 0:
        sp = audit_summary.get("shadow_portfolio") or {}
        st = (sp.get("strategies") or {}).get("ai") or {}
        bh = (sp.get("strategies") or {}).get("buy_hold_0050") or {}
        lines.extend([
            "",
            "## 五、累計實戰追蹤摘要",
            f"- 已結算週期樣本數：`{audit_summary['total_settled_horizons']}`",
            f"- 模擬部位累積報酬：AI 動態策略 `{st.get('total_return_pct', 0.0):+.2f}%` vs. 買進持有 0050 `{bh.get('total_return_pct', 0.0):+.2f}%`",
        ])

    return "\n".join(lines) + "\n"


def write_weekly_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
