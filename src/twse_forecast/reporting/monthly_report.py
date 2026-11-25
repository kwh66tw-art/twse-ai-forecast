"""月度實證績效評估報告（Markdown）產生器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from twse_forecast.config import HORIZONS
from twse_forecast.evaluation.calibration import render_calibration_diagnostics_markdown


def render_monthly_audit_report(
    *,
    audit_month: str,
    audit_summary: dict[str, Any],
    ledger_valid: bool,
    ledger_issues: list[str],
    recent_settlements: list[dict[str, Any]],
    calibration_profile: dict[str, Any] | None = None,
) -> str:
    """產出月度績效評估與 AI 能力實證稽核報告。"""
    sp = audit_summary.get("shadow_portfolio") or {}
    strats = sp.get("strategies") or {}
    ai_s = strats.get("ai") or {}
    q_s = strats.get("quant") or {}
    bh_s = strats.get("buy_hold_0050") or {}
    bal_s = strats.get("balanced_60_40") or {}

    integrity_text = "通過（SHA-256 雜湊鏈完整無竄改）" if ledger_valid else f"警示：{'; '.join(ledger_issues)}"

    lines: list[str] = [
        f"# 台股多週期預測系統 — 月度實證績效評估報告（{audit_month}）",
        "",
        f"- **報告產生時間**：`{audit_summary['generated_at']}`",
        f"- **帳本完整性稽核**：**{integrity_text}**",
        f"- **累計已鎖定週報數**：`{audit_summary['total_weekly_records']}` 週 ｜ **已到期結算預測數**：`{audit_summary['total_settled_horizons']}` 筆",
        "",
        "## 一、核心問題驗證：AI 模型是否真的創造正價值？",
    ]

    if audit_summary["total_settled_horizons"] == 0:
        lines.extend([
            "- 目前累積週數尚未滿最短預測週期（`1–4 週` 須滿 20 個交易日後始得正式結算），尚未產生已到期樣本。",
            "- 本月報告先行紀錄模擬建倉基準點與長週期軌跡通道狀態，待第 4 週起陸續到期後呈現完整 Brier Score 與 Alpha 成績單。",
        ])
    else:
        lines.extend([
            "| 預測週期 | 結算樣本 | 大盤自然上漲率 | AI 命中率 | 量化命中率 | AI Brier Score | 量化 Brier Score | 無腦看多 Brier | AI 淨精度改善 | 80% 區間涵蓋率 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for spec in HORIZONS:
            h = audit_summary["by_horizon"].get(spec.key) or {}
            if h.get("settled_count", 0) == 0:
                lines.append(f"| {spec.label} | 0（尚未到期） | - | - | - | - | - | - | - | - |")
                continue
            lines.append(
                f"| **{spec.label}** | {h['settled_count']} | {h['market_up_base_rate_pct']:.1f}% | "
                f"**{h['ai_hit_rate_pct']:.1f}%** | {h['quant_hit_rate_pct']:.1f}% | "
                f"`{h['ai_brier_score']:.4f}` | `{h['quant_brier_score']:.4f}` | `{h['naive_bull_brier_score']:.4f}` | "
                f"`{h['ai_brier_value_add']:+.4f}` | {h['ai_ci80_coverage_pct']:.1f}% |"
            )
        lines.extend([
            "",
            "> **指標解讀**：Brier Score 越低越精準（三分類純隨機猜測基準為 `0.6667`）；「AI 淨精度改善」為 `量化 Brier - AI Brier`，正值代表 AI 辯證推理確實優於純數學公式。",
        ])

    lines.extend([
        "",
        "## 二、模擬投資組合實戰績效（對標 0050 含息報酬，已扣除真實手續費與證交稅）",
    ])
    if not sp.get("available"):
        lines.append(f"- 模擬組合尚無法計算：{sp.get('reason')}")
    else:
        if sp.get("note"):
            lines.append(f"- 註記：{sp['note']}")
        lines.extend([
            "| 策略名稱 | 期末淨值（初始 100 萬） | 累積報酬率 | 年化夏普值 | 最大回檔幅度 | 累計摩擦成本 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| **AI 動態部位策略** | `{ai_s.get('equity', 1000000):,.0f}` | **`{ai_s.get('total_return_pct', 0.0):+.2f}%`** | `{ai_s.get('sharpe', 0.0):.2f}` | `{ai_s.get('max_drawdown_pct', 0.0):.2f}%` | `{ai_s.get('total_cost', 0.0):,.0f}` |",
            f"| 純量化動態策略 | `{q_s.get('equity', 1000000):,.0f}` | `{q_s.get('total_return_pct', 0.0):+.2f}%` | `{q_s.get('sharpe', 0.0):.2f}` | `{q_s.get('max_drawdown_pct', 0.0):.2f}%` | `{q_s.get('total_cost', 0.0):,.0f}` |",
            f"| 100% 買進持有 0050 | `{bh_s.get('equity', 1000000):,.0f}` | `{bh_s.get('total_return_pct', 0.0):+.2f}%` | `{bh_s.get('sharpe', 0.0):.2f}` | `{bh_s.get('max_drawdown_pct', 0.0):.2f}%` | `{bh_s.get('total_cost', 0.0):,.0f}` |",
            f"| 60/40 固定配置基準 | `{bal_s.get('equity', 1000000):,.0f}` | `{bal_s.get('total_return_pct', 0.0):+.2f}%` | `{bal_s.get('sharpe', 0.0):.2f}` | `{bal_s.get('max_drawdown_pct', 0.0):.2f}%` | `{bal_s.get('total_cost', 0.0):,.0f}` |",
        ])

    open_traj = audit_summary.get("open_trajectories") or []
    lines.extend(["", "## 三、未到期中長週期（含 1–3 年）階段性軌跡追蹤"])
    if not open_traj:
        lines.append("- 目前無進行中超過 5 個交易日之未到期追蹤紀錄。")
    else:
        lines.extend([
            "| 週次 | 週期 | 經過進度 | 目前報酬 | 時間縮放 80% 通道 | 是否在通道內 | 最大有利/不利偏離 (MFE/MAE) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for t in open_traj[-15:]:
            in_txt = "是" if t["in_scaled_ci80_corridor"] else "偏離"
            lines.append(
                f"| `{t['record_id']}` | {t['horizon_label']} | {t['progress_pct']:.1f}% ({t['elapsed_trading_days']}/{t['anchor_trading_days']}日) | "
                f"`{t['current_return_pct']:+.2f}%` | `{t['corridor_low']:,.0f} ~ {t['corridor_high']:,.0f}` | {in_txt} | "
                f"`{t['mfe_pct']:+.2f}%` / `{t['mae_pct']:+.2f}%` |"
            )

    if recent_settlements:
        lines.extend(["", "## 四、近期已到期結算明細（最新 15 筆）"])
        lines.extend([
            "| 週次 | 週期 | 基準日 → 結算日 | 實際報酬 | 實際分類 | AI 預測分類 (命中) | AI Brier | 量化 Brier |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for s in recent_settlements[-15:]:
            lines.append(
                f"| `{s['record_id']}` | {s['horizon_label']} | {s['reference_trade_date']} → {s['target_trade_date']} | "
                f"`{s['actual_return']*100:+.2f}%` | `{s['actual_outcome']}` | "
                f"`{s['ai_pred_class']}` ({'✓' if s['ai_hit'] else '✗'}) | `{s['ai_brier']:.4f}` | `{s['quant_brier']:.4f}` |"
            )

    if calibration_profile and calibration_profile.get("total_settled", 0) > 0:
        lines.extend(["", "## 五、閉環自我校準與動態授權韁繩狀態"])
        lines.extend(render_calibration_diagnostics_markdown(calibration_profile))

    return "\n".join(lines) + "\n"


def write_monthly_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
