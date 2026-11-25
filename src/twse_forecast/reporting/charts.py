"""高解析度台灣中文視覺化績效追蹤圖表產生器（180 DPI PNG，供儀表板與 Google Drive 原生預覽）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from twse_forecast.config import HORIZONS  # noqa: E402

BG_DARK = "#0b1120"
CARD_DARK = "#131c31"
BORDER_DARK = "#23304c"
TEXT_LIGHT = "#e2e8f0"
TEXT_MUTED = "#94a3b8"
COLOR_UP = "#10b981"
COLOR_FLAT = "#f59e0b"
COLOR_DOWN = "#ef4444"
COLOR_ACCENT = "#38bdf8"


def _configure_cjk_font() -> str:
    """自動偵測並套用可用的正體中文字型，避免圖表文字出現方框亂碼。"""
    candidates = [
        "PingFang TC",
        "Heiti TC",
        "STHeiti",
        "Noto Sans CJK TC",
        "Noto Sans TC",
        "Microsoft JhengHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available_names = {f.name for f in fm.fontManager.ttflist}
    chosen = "DejaVu Sans"
    for cand in candidates:
        if cand in available_names:
            chosen = cand
            break
    else:
        for f in fm.fontManager.ttflist:
            lower = f.name.lower()
            if "noto sans cjk" in lower or "pingfang" in lower or "heiti" in lower:
                chosen = f.name
                break

    plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def _style_axis(ax: Any) -> None:
    ax.set_facecolor(CARD_DARK)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BORDER_DARK)
    ax.grid(True, linestyle="--", alpha=0.25, color="#475569")


def _compute_drawdown_pct(values: list[float]) -> list[float]:
    if not values:
        return []
    peak = values[0]
    out: list[float] = []
    for v in values:
        if v > peak:
            peak = v
        dd = ((v / peak) - 1.0) * 100.0 if peak > 0 else 0.0
        out.append(dd)
    return out


def generate_horizons_chart(record: dict[str, Any], out_path: Path) -> Path:
    """繪製五週期多／盤／空機率分布與 80% 信賴點位通道／失效關鍵位雙聯圖。"""
    _configure_cjk_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.0), dpi=180)
    fig.patch.set_facecolor(BG_DARK)

    labels = [spec.label for spec in HORIZONS][::-1]
    keys = [spec.key for spec in HORIZONS][::-1]
    ai_h = record["ai_forecast"]["horizons"]
    ref_close = float(record["reference_taiex_close"])
    rec_id = record["record_id"]
    ref_date = record["reference_trade_date"]

    # 左圖：三分類機率水平堆疊條
    _style_axis(ax1)
    p_ups = [float(ai_h[k]["prob_up"]) * 100.0 for k in keys]
    p_flats = [float(ai_h[k]["prob_flat"]) * 100.0 for k in keys]
    p_downs = [float(ai_h[k]["prob_down"]) * 100.0 for k in keys]

    y_pos = list(range(len(keys)))
    ax1.barh(y_pos, p_ups, color=COLOR_UP, height=0.52, label="上漲機率")
    ax1.barh(y_pos, p_flats, left=p_ups, color=COLOR_FLAT, height=0.52, label="盤整機率")
    left_down = [u + f for u, f in zip(p_ups, p_flats, strict=True)]
    ax1.barh(y_pos, p_downs, left=left_down, color=COLOR_DOWN, height=0.52, label="下跌機率")

    for i, (u, f, d) in enumerate(zip(p_ups, p_flats, p_downs, strict=True)):
        if u >= 12:
            ax1.text(u / 2, i, f"{u:.1f}%", va="center", ha="center", color="#06281e", fontsize=9, fontweight="bold")
        if f >= 12:
            ax1.text(u + f / 2, i, f"{f:.1f}%", va="center", ha="center", color="#2c1900", fontsize=9, fontweight="bold")
        if d >= 12:
            ax1.text(u + f + d / 2, i, f"{d:.1f}%", va="center", ha="center", color="#ffffff", fontsize=9, fontweight="bold")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, color=TEXT_LIGHT, fontsize=10, fontweight="bold")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.95, len(keys) - 0.35)
    ax1.set_xlabel("機率分布 (%)", color=TEXT_MUTED, fontsize=10)
    ax1.set_title("各週期多／盤／空機率分布", color=TEXT_LIGHT, fontsize=12, fontweight="bold", pad=10)
    ax1.legend(loc="lower right", ncol=3, facecolor=BG_DARK, edgecolor=BORDER_DARK, labelcolor=TEXT_LIGHT, fontsize=8.5)

    # 右圖：80% 信賴點位區間 + 預期目標 + 失效關鍵點位
    _style_axis(ax2)
    lows = [float(ai_h[k]["ci80_low_level"]) for k in keys]
    highs = [float(ai_h[k]["ci80_high_level"]) for k in keys]
    exps = [ref_close * (1.0 + float(ai_h[k]["expected_return"])) for k in keys]
    invs = [float(ai_h[k]["invalidation_level"]) for k in keys]

    ax2.axvline(ref_close, color="#e2e8f0", linestyle="--", linewidth=1.4, alpha=0.85, label=f"基準點位 {ref_close:,.0f}")
    for i, (lo, hi, ex, inv) in enumerate(zip(lows, highs, exps, invs, strict=True)):
        ax2.plot([lo, hi], [i, i], color=COLOR_ACCENT, linewidth=7, alpha=0.65, solid_capstyle="round")
        ax2.scatter([ex], [i], color="#ffffff", s=55, zorder=5, label="預期目標點位" if i == 0 else "")
        ax2.scatter([inv], [i], color=COLOR_DOWN, marker="X", s=70, zorder=6, label="失效關鍵點位" if i == 0 else "")
        ax2.text(hi + (max(highs) - min(lows)) * 0.015, i, f"{lo:,.0f}~{hi:,.0f}", va="center", color=TEXT_MUTED, fontsize=8)

    min_x = min(min(lows), min(invs), ref_close) * 0.96
    max_x = max(highs) * 1.15
    ax2.set_xlim(min_x, max_x)
    ax2.set_ylim(-1.05, len(keys) - 0.35)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, color=TEXT_LIGHT, fontsize=10, fontweight="bold")
    ax2.set_xlabel("台灣加權股價指數點位 (TAIEX)", color=TEXT_MUTED, fontsize=10)
    ax2.set_title("80% 信賴點位通道與失效防線", color=TEXT_LIGHT, fontsize=12, fontweight="bold", pad=10)
    ax2.legend(loc="lower right", ncol=3, facecolor=BG_DARK, edgecolor=BORDER_DARK, labelcolor=TEXT_LIGHT, fontsize=8.2)

    fig.suptitle(
        f"台股大盤五週期 AI 預測總覽（{rec_id} ｜ 基準日 {ref_date} ｜ 加權指數 {ref_close:,.2f}）",
        color=TEXT_LIGHT,
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=BG_DARK)
    plt.close(fig)
    return out_path


def generate_performance_chart(audit_summary: dict[str, Any], out_path: Path) -> Path:
    """繪製模擬投資組合淨值曲線、最大回檔保護與 AI 建議持股水位三合一追蹤圖。"""
    _configure_cjk_font()
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(13, 8.8), dpi=180, sharex=True, gridspec_kw={"height_ratios": [2.2, 1.3, 1.0]}
    )
    fig.patch.set_facecolor(BG_DARK)
    for ax in (ax1, ax2, ax3):
        _style_axis(ax)

    sp = audit_summary.get("shadow_portfolio") or {}
    curve = sp.get("curve") or []

    if not curve:
        ax1.text(0.5, 0.5, "尚無模擬投資組合資料", ha="center", va="center", color=TEXT_MUTED, fontsize=12, transform=ax1.transAxes)
    else:
        labels = [c["label"] for c in curve]
        x = list(range(len(labels)))
        ai_eq = [c["ai_equity"] / 10000.0 for c in curve]
        q_eq = [c["quant_equity"] / 10000.0 for c in curve]
        bh_eq = [c["buy_hold_equity"] / 10000.0 for c in curve]
        bal_eq = [c["balanced_60_40_equity"] / 10000.0 for c in curve]
        allocs = [c["ai_alloc_pct"] for c in curve]

        if len(curve) == 1:
            ax1.scatter(x, ai_eq, color=COLOR_UP, s=90, zorder=5, label="AI 動態策略 (100.0 萬基期)")
            ax1.text(
                0.5,
                0.76,
                "基期建倉完成（100.0 萬元起始點）\n將於次週收盤結算後展開實戰淨值對照曲線",
                ha="center",
                va="center",
                color=TEXT_MUTED,
                fontsize=11,
                transform=ax1.transAxes,
            )
            ax1.set_ylim(95, 105)
        else:
            ax1.plot(x, ai_eq, color=COLOR_UP, linewidth=2.6, marker="o", markersize=4, label="AI 動態策略")
            ax1.plot(x, q_eq, color=COLOR_ACCENT, linewidth=1.8, label="純量化基準")
            ax1.plot(x, bh_eq, color="#94a3b8", linewidth=1.8, linestyle="--", label="100% 買進持有 0050")
            ax1.plot(x, bal_eq, color=COLOR_FLAT, linewidth=1.5, linestyle=":", label="60/40 固定配置")

        ax1.set_ylabel("資產淨值（新台幣萬元）", color=TEXT_MUTED, fontsize=10)
        ax1.set_title("模擬投資組合淨值走勢（對標 0050 含息報酬，已扣手續費與證交稅）", color=TEXT_LIGHT, fontsize=12, fontweight="bold")
        ax1.legend(loc="upper left", facecolor=BG_DARK, edgecolor=BORDER_DARK, labelcolor=TEXT_LIGHT, fontsize=8.5)

        # 中圖：回檔幅度 (%)
        dd_ai = _compute_drawdown_pct(ai_eq)
        dd_bh = _compute_drawdown_pct(bh_eq)
        ax2.fill_between(x, dd_ai, 0, color=COLOR_UP, alpha=0.25, label="AI 策略回檔 (%)")
        ax2.plot(x, dd_ai, color=COLOR_UP, linewidth=1.6)
        ax2.plot(x, dd_bh, color="#94a3b8", linewidth=1.4, linestyle="--", label="0050 大盤回檔 (%)")
        ax2.set_ylabel("回檔幅度 (%)", color=TEXT_MUTED, fontsize=10)
        ax2.set_title("最大回檔保護對照（Underwater Drawdown）", color=TEXT_LIGHT, fontsize=11, fontweight="bold")
        ax2.legend(loc="lower left", facecolor=BG_DARK, edgecolor=BORDER_DARK, labelcolor=TEXT_LIGHT, fontsize=8.5)

        # 下圖：AI 建議持股水位柱狀圖
        ax3.bar(x, allocs, color=COLOR_ACCENT, width=0.5, alpha=0.85)
        for xi, al in zip(x, allocs, strict=True):
            ax3.text(xi, al + 2.5, f"{al:.0f}%", ha="center", va="bottom", color=TEXT_LIGHT, fontsize=8.5)
        ax3.set_ylim(0, 115)
        ax3.set_ylabel("持股水位 (%)", color=TEXT_MUTED, fontsize=10)
        ax3.set_title("每週 AI 建議模擬持股水位（0050.TW 曝險比例）", color=TEXT_LIGHT, fontsize=11, fontweight="bold")

        step = max(1, len(labels) // 10)
        ticks = list(range(0, len(labels), step))
        ax3.set_xticks(ticks)
        ax3.set_xticklabels([labels[i] for i in ticks], color=TEXT_LIGHT, fontsize=9)

    fig.suptitle(
        f"台股模擬投資組合績效與風險追蹤（累計週數：{audit_summary.get('total_weekly_records', 0)} 週）",
        color=TEXT_LIGHT,
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=BG_DARK)
    plt.close(fig)
    return out_path


def generate_scorecard_chart(
    audit_summary: dict[str, Any],
    latest_record: dict[str, Any] | None,
    out_path: Path,
) -> Path:
    """繪製 AI 預測準確度（Brier Score／命中率）計分卡；尚未到期時顯示各週期到期進度與分數差異。"""
    _configure_cjk_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=180)
    fig.patch.set_facecolor(BG_DARK)
    _style_axis(ax1)
    _style_axis(ax2)

    by_h = audit_summary.get("by_horizon") or {}
    settled_specs = [spec for spec in HORIZONS if (by_h.get(spec.key) or {}).get("settled_count", 0) > 0]

    if not settled_specs:
        # 尚無正式結算樣本時：左圖展示各週期結算門檻倒數、右圖展示最新期 AI vs 純量化分數對照
        labels = [spec.label for spec in HORIZONS]
        anchors = [spec.anchor_trading_days for spec in HORIZONS]
        y_pos = list(range(len(labels)))
        ax1.barh(y_pos, anchors, color="#334155", height=0.5, label="正式結算所需交易日數")
        for i, d in enumerate(anchors):
            ax1.text(d + 6, i, f"滿 {d} 交易日結算", va="center", color=TEXT_LIGHT, fontsize=9)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels, color=TEXT_LIGHT, fontsize=10)
        ax1.set_xlim(0, 460)
        ax1.set_xlabel("交易日門檻", color=TEXT_MUTED, fontsize=10)
        ax1.set_title("五週期結算驗算進度（累積滿第 20 交易日起陸續開獎）", color=TEXT_LIGHT, fontsize=11, fontweight="bold")

        if latest_record:
            q_scores = [float(latest_record["quant_baseline"]["horizons"][s.key]["quant_score"]) for s in HORIZONS]
            ai_scores = [float(latest_record["ai_forecast"]["horizons"][s.key]["ai_score"]) for s in HORIZONS]
            x = list(range(len(labels)))
            w = 0.36
            ax2.bar([i - w / 2 for i in x], q_scores, width=w, color=COLOR_ACCENT, label="純量化基準分數")
            ax2.bar([i + w / 2 for i in x], ai_scores, width=w, color=COLOR_UP, label="AI 辯證分數 (±25 受限)")
            ax2.axhline(0, color=TEXT_MUTED, linewidth=1.0)
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, color=TEXT_LIGHT, fontsize=9.5)
            ax2.set_ylim(-105, 105)
            ax2.set_ylabel("多空評分 (-100 ~ +100)", color=TEXT_MUTED, fontsize=10)
            ax2.set_title("最新週次：AI 辯證 vs. 純量化基準分數比較", color=TEXT_LIGHT, fontsize=11, fontweight="bold")
            ax2.legend(loc="upper left", facecolor=BG_DARK, edgecolor=BORDER_DARK, labelcolor=TEXT_LIGHT, fontsize=8.5)
        else:
            ax2.text(0.5, 0.5, "尚無預測紀錄", ha="center", va="center", color=TEXT_MUTED, transform=ax2.transAxes)
    else:
        labels = [s.label for s in settled_specs]
        x = list(range(len(settled_specs)))
        w = 0.25

        ai_briers = [by_h[s.key]["ai_brier_score"] for s in settled_specs]
        q_briers = [by_h[s.key]["quant_brier_score"] for s in settled_specs]
        nb_briers = [by_h[s.key]["naive_bull_brier_score"] for s in settled_specs]

        ax1.bar([i - w for i in x], ai_briers, width=w, color=COLOR_UP, label="AI 模型")
        ax1.bar(x, q_briers, width=w, color=COLOR_ACCENT, label="純量化基準")
        ax1.bar([i + w for i in x], nb_briers, width=w, color="#64748b", label="無腦看多基準")
        ax1.axhline(0.6667, color=COLOR_DOWN, linestyle="--", linewidth=1.2, label="純隨機猜測 (0.667)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, color=TEXT_LIGHT, fontsize=9.5)
        ax1.set_ylabel("Brier Score（越低越準）", color=TEXT_MUTED, fontsize=10)
        ax1.set_title("機率校準誤差對照（Brier Score）", color=TEXT_LIGHT, fontsize=11, fontweight="bold")
        ax1.legend(loc="upper right", facecolor=BG_DARK, edgecolor=BORDER_DARK, labelcolor=TEXT_LIGHT, fontsize=8)

        ai_hits = [by_h[s.key]["ai_hit_rate_pct"] for s in settled_specs]
        q_hits = [by_h[s.key]["quant_hit_rate_pct"] for s in settled_specs]
        base_ups = [by_h[s.key]["market_up_base_rate_pct"] for s in settled_specs]

        ax2.bar([i - w for i in x], ai_hits, width=w, color=COLOR_UP, label="AI 命中率 (%)")
        ax2.bar(x, q_hits, width=w, color=COLOR_ACCENT, label="量化命中率 (%)")
        ax2.bar([i + w for i in x], base_ups, width=w, color=COLOR_FLAT, label="大盤自然上漲率 (%)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, color=TEXT_LIGHT, fontsize=9.5)
        ax2.set_ylim(0, 110)
        ax2.set_ylabel("百分比 (%)", color=TEXT_MUTED, fontsize=10)
        ax2.set_title("方向命中率 vs. 市場基準率", color=TEXT_LIGHT, fontsize=11, fontweight="bold")
        ax2.legend(loc="upper right", facecolor=BG_DARK, edgecolor=BORDER_DARK, labelcolor=TEXT_LIGHT, fontsize=8)

    fig.suptitle(
        f"AI 能力實證計分卡（累計已到期結算：{audit_summary.get('total_settled_horizons', 0)} 筆）",
        color=TEXT_LIGHT,
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=BG_DARK)
    plt.close(fig)
    return out_path


def generate_all_charts(
    *,
    charts_dir: Path,
    records: list[dict[str, Any]],
    audit_summary: dict[str, Any],
) -> dict[str, Path]:
    """一次產出全部三張高解析度視覺化圖表並回傳路徑字典。"""
    charts_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    latest = records[-1] if records else None
    if latest:
        outputs["horizons"] = generate_horizons_chart(latest, charts_dir / "horizons_forecast.png")
    outputs["performance"] = generate_performance_chart(
        audit_summary, charts_dir / "performance_tracking.png"
    )
    outputs["scorecard"] = generate_scorecard_chart(
        audit_summary, latest, charts_dir / "ai_accuracy_scorecard.png"
    )
    return outputs
