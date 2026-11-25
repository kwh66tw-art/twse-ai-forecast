"""單檔離線互動式 HTML 儀表板產生器（內嵌 SVG 圖表，免外部 CDN）。"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any

from twse_forecast.config import HORIZONS, TAIPEI_TZ


def _format_taipei_ts(raw_iso: str | None) -> str:
    """將 ISO 時間字串統一格式化為易讀之台灣時間 YYYY-MM-DD HH:MM:SS (UTC+8)。"""
    if not raw_iso or raw_iso == "-":
        return "-"
    try:
        dt = datetime.fromisoformat(raw_iso).astimezone(TAIPEI_TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S (UTC+8)")
    except ValueError:
        return raw_iso


def _build_equity_svg(curve: list[dict[str, Any]]) -> str:
    """以原生 SVG 繪製模擬投資組合淨值走勢對照圖。"""
    if len(curve) < 2:
        return (
            '<div class="empty-chart">目前累積僅有基期建倉點，將於下週起繪製 AI 動態策略 vs. 0050 大盤實戰淨值曲線。</div>'
        )
    w, h = 760, 240
    pad_l, pad_r, pad_t, pad_b = 64, 24, 22, 34
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    all_vals: list[float] = []
    for pt in curve:
        all_vals.extend([pt["ai_equity"], pt["quant_equity"], pt["buy_hold_equity"]])
    min_v = min(all_vals) * 0.995
    max_v = max(all_vals) * 1.005
    if max_v <= min_v:
        max_v = min_v + 10000.0

    def to_xy(idx: int, val: float) -> tuple[float, float]:
        x = pad_l + (idx / (len(curve) - 1)) * plot_w
        y = pad_t + plot_h - ((val - min_v) / (max_v - min_v)) * plot_h
        return round(x, 1), round(y, 1)

    def polyline(key: str, color: str, width: float = 2.4, dash: str = "") -> str:
        pts = " ".join(f"{x},{y}" for x, y in (to_xy(i, c[key]) for i, c in enumerate(curve)))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline fill="none" stroke="{color}" stroke-width="{width}"{dash_attr} points="{pts}" />'

    grid_lines: list[str] = []
    for step in range(5):
        frac = step / 4.0
        val = min_v + frac * (max_v - min_v)
        y = round(pad_t + plot_h - frac * plot_h, 1)
        grid_lines.append(f'<line x1="{pad_l}" y1="{y}" x2="{w - pad_r}" y2="{y}" stroke="#273248" stroke-width="1" />')
        grid_lines.append(
            f'<text x="{pad_l - 8}" y="{y + 4}" text-anchor="end" fill="#8b9bb4" font-size="11">{val/10000:.1f}萬</text>'
        )

    first_lbl = html.escape(curve[0]["label"])
    last_lbl = html.escape(curve[-1]["label"])

    return (
        f'<svg viewBox="0 0 {w} {h}" class="equity-svg" role="img" aria-label="模擬淨值曲線">'
        + "".join(grid_lines)
        + polyline("buy_hold_equity", "#64748b", 2.0, "5,4")
        + polyline("quant_equity", "#38bdf8", 2.2)
        + polyline("ai_equity", "#10b981", 3.0)
        + f'<text x="{pad_l}" y="{h - 10}" fill="#8b9bb4" font-size="11">{first_lbl}</text>'
        + f'<text x="{w - pad_r}" y="{h - 10}" text-anchor="end" fill="#8b9bb4" font-size="11">{last_lbl}</text>'
        + "</svg>"
    )


def render_dashboard_html(
    *,
    records: list[dict[str, Any]],
    latest_features: dict[str, Any] | None,
    audit_summary: dict[str, Any] | None,
    ledger_valid: bool,
    ledger_issues: list[str],
) -> str:
    """產生完整單檔深色金融風格儀表板 HTML。"""
    latest = records[-1] if records else None
    sp = (audit_summary or {}).get("shadow_portfolio") or {}
    strats = sp.get("strategies") or {}
    ai_s = strats.get("ai") or {}
    bh_s = strats.get("buy_hold_0050") or {}
    q_s = strats.get("quant") or {}
    curve = sp.get("curve") or []

    badge_cls = "badge-ok" if ledger_valid else "badge-err"
    badge_txt = (
        f"SHA-256 雜湊鏈驗證通過（{len(records)} 筆鎖定紀錄）"
        if ledger_valid
        else html.escape(f"帳本異常：{'; '.join(ledger_issues)}")
    )

    if latest:
        ai_f = latest["ai_forecast"]
        q_b = latest["quant_baseline"]
        ref_close = float(latest["reference_taiex_close"])
        latest_title = html.escape(f"{latest['record_id']} ｜ 基準日 {latest['reference_trade_date']} ｜ 加權指數 {ref_close:,.2f}")
        alloc_pct = f"{ai_f['target_allocation'] * 100:.1f}%"
        q_alloc_pct = f"{q_b['target_allocation'] * 100:.1f}%"
        model_label = html.escape(ai_f["model_name"])
    else:
        latest_title = "尚無預測紀錄"
        alloc_pct = "-"
        q_alloc_pct = "-"
        model_label = "-"

    horizon_cards_html: list[str] = []
    if latest:
        for spec in HORIZONS:
            af = latest["ai_forecast"]["horizons"][spec.key]
            qf = latest["quant_baseline"]["horizons"][spec.key]
            pu = af["prob_up"] * 100.0
            pf = af["prob_flat"] * 100.0
            pd = af["prob_down"] * 100.0
            score_cls = "pos" if af["ai_score"] >= 10 else ("neg" if af["ai_score"] <= -10 else "neu")
            horizon_cards_html.append(
                f"""
                <div class="card horizon-card">
                  <div class="h-top">
                    <span class="h-title">{html.escape(spec.label)}</span>
                    <span class="h-anchor">{spec.anchor_trading_days} 交易日</span>
                  </div>
                  <div class="h-score-row">
                    <span class="score {score_cls}">AI {af['ai_score']:+.1f}</span>
                    <span class="q-meta">量化 {qf['quant_score']:+.1f} (Δ {af['delta_score']:+.1f})</span>
                  </div>
                  <div class="prob-bar" title="上漲 {pu:.1f}% / 盤整 {pf:.1f}% / 下跌 {pd:.1f}%">
                    <div class="seg-up" style="width:{pu:.2f}%"></div>
                    <div class="seg-flat" style="width:{pf:.2f}%"></div>
                    <div class="seg-down" style="width:{pd:.2f}%"></div>
                  </div>
                  <div class="prob-legend">
                    <span class="c-up">漲 {pu:.1f}%</span>
                    <span class="c-flat">盤 {pf:.1f}%</span>
                    <span class="c-down">跌 {pd:.1f}%</span>
                  </div>
                  <div class="h-levels">
                    <div>預期報酬：<strong>{af['expected_return']*100:+.2f}%</strong></div>
                    <div>80% 區間：<code>{af['ci80_low_level']:,.0f} ~ {af['ci80_high_level']:,.0f}</code></div>
                    <div>失效關鍵位：<code>{af['invalidation_level']:,.0f}</code></div>
                  </div>
                  <p class="h-thesis">{html.escape(af['thesis'])}</p>
                </div>
                """
            )

    brier_rows_html: list[str] = []
    by_h = (audit_summary or {}).get("by_horizon") or {}
    for spec in HORIZONS:
        row = by_h.get(spec.key) or {}
        if row.get("settled_count", 0) == 0:
            brier_rows_html.append(
                f"<tr><td>{html.escape(spec.label)}</td><td>0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>"
            )
        else:
            val_add = row["ai_brier_value_add"]
            va_cls = "pos" if val_add > 0 else ("neg" if val_add < 0 else "neu")
            brier_rows_html.append(
                f"<tr>"
                f"<td>{html.escape(spec.label)}</td>"
                f"<td>{row['settled_count']}</td>"
                f"<td>{row['ai_hit_rate_pct']:.1f}%</td>"
                f"<td>{row['quant_hit_rate_pct']:.1f}%</td>"
                f"<td>{row['ai_brier_score']:.4f}</td>"
                f"<td>{row['quant_brier_score']:.4f}</td>"
                f"<td class='{va_cls}'>{val_add:+.4f}</td>"
                f"</tr>"
            )

    debate_html = ""
    if latest:
        ai_f = latest["ai_forecast"]
        debate_html = f"""
        <section class="card debate-section">
          <h2>本週 AI 多空辯證與投資長裁決</h2>
          <div class="debate-grid">
            <div class="debate-box bull"><h3>多頭論點（Bull Case）</h3><p>{html.escape(ai_f['bull_case'])}</p></div>
            <div class="debate-box bear"><h3>空頭稽核（Bear Case）</h3><p>{html.escape(ai_f['bear_case'])}</p></div>
            <div class="debate-box cio"><h3>投資長裁決（CIO Synthesis）</h3><p>{html.escape(ai_f['cio_synthesis'])}</p></div>
            <div class="debate-box diff"><h3>與純量化差異（Override Rationale）</h3><p>{html.escape(ai_f['override_rationale'])}</p></div>
          </div>
        </section>
        """

    micro_panel_html = ""
    if latest_features:
        cm = latest_features.get("cross_market") or {}
        fn = latest_features.get("fundamentals_valuation") or {}
        mc = latest_features.get("microstructure_flows") or {}
        br = latest_features.get("breadth") or {}
        st = latest_features.get("sentiment") or {}
        micro_panel_html = f"""
        <section class="card">
          <h2>五大維度核心指標快照</h2>
          <div class="metrics-grid">
            <div class="metric-item"><span class="m-lbl">台積電 ADR 溢價率</span><span class="m-val">{cm.get('tsmc_adr_premium_pct')}%</span></div>
            <div class="metric-item"><span class="m-lbl">上市中位數 P/B ｜ ERP</span><span class="m-val">{fn.get('market_median_pb')} ｜ {fn.get('equity_risk_premium_pct')}%</span></div>
            <div class="metric-item"><span class="m-lbl">月營收正成長廣度</span><span class="m-val">{fn.get('monthly_revenue_pos_ratio_pct')}%</span></div>
            <div class="metric-item"><span class="m-lbl">外資 5 日現貨淨買超</span><span class="m-val">{mc.get('foreign_5d_net_b')} 億</span></div>
            <div class="metric-item"><span class="m-lbl">外資台指期淨未平倉</span><span class="m-val">{mc.get('foreign_tx_net_oi')} 口</span></div>
            <div class="metric-item"><span class="m-lbl">選擇權 Put/Call Ratio</span><span class="m-val">{mc.get('pc_ratio_latest_pct')}%</span></div>
            <div class="metric-item"><span class="m-lbl">30 檔籃子站上季線</span><span class="m-val">{br.get('pct_above_ma60')}%</span></div>
            <div class="metric-item"><span class="m-lbl">PTT 股板狂熱 / 反向</span><span class="m-val">{st.get('retail_euphoria_score', '-')} / {st.get('contrarian_signal', '-')}</span></div>
          </div>
        </section>
        """

    ledger_rows_html: list[str] = []
    for r in reversed(records[-12:]):
        mode = "補跑/基期" if r["is_late_catchup"] else "週一盤前"
        ledger_rows_html.append(
            f"<tr><td>{html.escape(r['record_id'])}</td>"
            f"<td>{html.escape(r['reference_trade_date'])}</td>"
            f"<td>{r['reference_taiex_close']:,.2f}</td>"
            f"<td>{r['ai_forecast']['target_allocation']*100:.1f}%</td>"
            f"<td>{mode}</td>"
            f"<td><code>{html.escape(r['record_hash'][:16])}...</code></td></tr>"
        )

    equity_svg = _build_equity_svg(curve)
    now_tw = datetime.now(TAIPEI_TZ)
    rendered_at_str = now_tw.strftime("%Y-%m-%d %H:%M:%S (UTC+8)")
    cache_ver = str(int(now_tw.timestamp()))
    forecast_locked_at = html.escape(_format_taipei_ts(latest["executed_at_iso"]) if latest else "尚無紀錄")
    audit_generated_at = html.escape(
        _format_taipei_ts((audit_summary or {}).get("generated_at")) if audit_summary else rendered_at_str
    )

    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>台股大盤多週期 AI 預測與實證儀表板</title>
<style>
  :root {{
    --bg: #0b1120;
    --card: #131c31;
    --border: #23304c;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --up: #10b981;
    --flat: #f59e0b;
    --down: #ef4444;
    --accent: #38bdf8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 24px;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", "Microsoft JhengHei", sans-serif;
    line-height: 1.55;
  }}
  .container {{ max-width: 1280px; margin: 0 auto; display: flex; flex-direction: column; gap: 18px; }}
  header {{
    display: flex; flex-direction: column; gap: 12px;
    padding: 18px 22px; background: var(--card); border: 1px solid var(--border); border-radius: 12px;
  }}
  .header-top {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; }}
  .stamp-bar {{
    display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center;
    padding-top: 10px; border-top: 1px solid var(--border); font-size: 12px; color: var(--muted);
  }}
  .stamp-item strong {{ color: #cbd5e1; font-weight: 600; }}
  .stamp-item a {{ color: var(--accent); text-decoration: none; }}
  .stamp-item a:hover {{ text-decoration: underline; }}
  h1 {{ margin: 0; font-size: 22px; font-weight: 700; }}
  h2 {{ margin: 0 0 12px 0; font-size: 16px; color: var(--accent); }}
  h3 {{ margin: 0 0 6px 0; font-size: 14px; }}
  .subtitle {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
  .badge {{ padding: 6px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
  .badge-ok {{ background: rgba(16, 185, 129, 0.16); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
  .badge-err {{ background: rgba(239, 68, 68, 0.16); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
  .kpi-strip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; }}
  .kpi-lbl {{ font-size: 12px; color: var(--muted); }}
  .kpi-val {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
  .kpi-sub {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
  .horizons-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; }}
  .h-top {{ display: flex; justify-content: space-between; align-items: baseline; }}
  .h-title {{ font-size: 16px; font-weight: 700; }}
  .h-anchor {{ font-size: 12px; color: var(--muted); }}
  .h-score-row {{ display: flex; justify-content: space-between; align-items: baseline; margin: 10px 0 8px 0; }}
  .score {{ font-size: 18px; font-weight: 700; }}
  .pos {{ color: var(--up); }} .neg {{ color: var(--down); }} .neu {{ color: var(--flat); }}
  .q-meta {{ font-size: 12px; color: var(--muted); }}
  .prob-bar {{ display: flex; height: 9px; border-radius: 999px; overflow: hidden; background: #1e293b; }}
  .seg-up {{ background: var(--up); }} .seg-flat {{ background: var(--flat); }} .seg-down {{ background: var(--down); }}
  .prob-legend {{ display: flex; justify-content: space-between; font-size: 11px; margin: 6px 0 10px 0; }}
  .c-up {{ color: var(--up); }} .c-flat {{ color: var(--flat); }} .c-down {{ color: var(--down); }}
  .h-levels {{ font-size: 12px; color: var(--text); display: flex; flex-direction: column; gap: 3px; border-top: 1px solid var(--border); padding-top: 8px; }}
  .h-thesis {{ font-size: 12px; color: var(--muted); margin: 8px 0 0 0; }}
  .split-grid {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 16px; }}
  @media (max-width: 960px) {{ .split-grid {{ grid-template-columns: 1fr; }} }}
  .equity-svg {{ width: 100%; height: auto; display: block; }}
  .chart-legend {{ display: flex; gap: 16px; font-size: 12px; color: var(--muted); margin-bottom: 6px; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }}
  .empty-chart {{ padding: 36px 16px; text-align: center; color: var(--muted); border: 1px dashed var(--border); border-radius: 8px; font-size: 13px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; font-size: 12px; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: #cbd5e1; }}
  .debate-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
  .debate-box {{ background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-size: 13px; }}
  .debate-box p {{ margin: 0; color: #cbd5e1; }}
  .debate-box.bull h3 {{ color: var(--up); }}
  .debate-box.bear h3 {{ color: var(--down); }}
  .debate-box.cio h3 {{ color: var(--accent); }}
  .debate-box.diff h3 {{ color: var(--flat); }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }}
  .metric-item {{ background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; display: flex; flex-direction: column; gap: 4px; }}
  .m-lbl {{ font-size: 11px; color: var(--muted); }}
  .m-val {{ font-size: 15px; font-weight: 600; }}
  .charts-gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; }}
  .chart-img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--border); display: block; }}
  .chart-caption {{ font-size: 12px; color: var(--muted); margin-top: 6px; }}
</style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-top">
        <div>
          <h1>台灣股市大盤多週期 AI 預測與實證儀表板</h1>
          <div class="subtitle">{latest_title} ｜ 執行模型：{model_label}</div>
        </div>
        <div class="badge {badge_cls}">{badge_txt}</div>
      </div>
      <div class="stamp-bar">
        <span class="stamp-item"><strong>預測資料鎖定：</strong>{forecast_locked_at}</span>
        <span class="stamp-item"><strong>績效稽核重算：</strong>{audit_generated_at}</span>
        <span class="stamp-item"><strong>網頁最後更新：</strong><span id="deploy-stamp">{rendered_at_str}</span></span>
      </div>
    </header>

    <section class="kpi-strip">
      <div class="card"><div class="kpi-lbl">建議模擬持股水位 (0050)</div><div class="kpi-val">{alloc_pct}</div><div class="kpi-sub">純量化基準：{q_alloc_pct}</div></div>
      <div class="card"><div class="kpi-lbl">AI 動態策略模擬報酬</div><div class="kpi-val">{ai_s.get('total_return_pct', 0.0):+.2f}%</div><div class="kpi-sub">0050 買進持有：{bh_s.get('total_return_pct', 0.0):+.2f}%</div></div>
      <div class="card"><div class="kpi-lbl">AI 策略夏普值 / 最大回檔</div><div class="kpi-val">{ai_s.get('sharpe', 0.0):.2f}</div><div class="kpi-sub">Max Drawdown：{ai_s.get('max_drawdown_pct', 0.0):.2f}%（Quant：{q_s.get('total_return_pct', 0.0):+.2f}%）</div></div>
      <div class="card"><div class="kpi-lbl">已鎖定預測週數 / 結算樣本</div><div class="kpi-val">{len(records)} 週 / {(audit_summary or {}).get('total_settled_horizons', 0)} 筆</div><div class="kpi-sub">含雙邊 ETF 手續費與證交稅</div></div>
    </section>

    <section>
      <h2>五週期量化與 AI 辯證預測總覽</h2>
      <div class="horizons-grid">
        {"".join(horizon_cards_html)}
      </div>
    </section>

    {debate_html}
    {micro_panel_html}

    <section class="split-grid">
      <div class="card">
        <h2>模擬投資組合淨值曲線（初始 100 萬元，對標 0050 含息）</h2>
        <div class="chart-legend">
          <span><span class="dot" style="background:#10b981"></span>AI 動態策略</span>
          <span><span class="dot" style="background:#38bdf8"></span>純量化基準</span>
          <span><span class="dot" style="background:#64748b"></span>100% 買進持有 0050</span>
        </div>
        {equity_svg}
      </div>
      <div class="card">
        <h2>各週期 Brier Score 與 AI 淨附加價值稽核</h2>
        <table>
          <thead><tr><th>週期</th><th>樣本</th><th>AI 命中</th><th>量化命中</th><th>AI Brier</th><th>量化 Brier</th><th>淨改善</th></tr></thead>
          <tbody>{"".join(brier_rows_html)}</tbody>
        </table>
      </div>
    </section>

    <section class="card">
      <h2>視覺化績效追蹤與預測通道圖表（180 DPI 高解析快照）</h2>
      <div class="charts-gallery">
        <div>
          <img class="chart-img" src="charts/horizons_forecast.png?v={cache_ver}" alt="五週期預測機率與點位通道圖" />
          <div class="chart-caption">五週期多／盤／空機率分布與 80% 信賴點位通道</div>
        </div>
        <div>
          <img class="chart-img" src="charts/performance_tracking.png?v={cache_ver}" alt="模擬投資組合淨值與回檔追蹤圖" />
          <div class="chart-caption">模擬投資組合淨值走勢、回檔保護與建議持股水位</div>
        </div>
        <div>
          <img class="chart-img" src="charts/ai_accuracy_scorecard.png?v={cache_ver}" alt="AI 能力實證計分卡圖" />
          <div class="chart-caption">AI 模型 vs. 純量化基準 Brier Score 與命中率計分卡</div>
        </div>
      </div>
    </section>

    <section class="card">
      <h2>不可竄改預測帳本存證紀錄（最近 12 週）</h2>
      <table>
        <thead><tr><th>週次</th><th>基準交易日</th><th>TAIEX 收盤</th><th>建議持股</th><th>時段性質</th><th>SHA-256 雜湊</th></tr></thead>
        <tbody>{"".join(ledger_rows_html)}</tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""


def write_dashboard_html(path: Path, html_content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_content, encoding="utf-8")
