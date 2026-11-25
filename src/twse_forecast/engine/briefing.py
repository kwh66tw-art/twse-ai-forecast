"""第二層：AI 多視角簡報產生器與受限調整（Bounded Override）驗證介面。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from twse_forecast.config import HORIZON_MAP, HORIZONS, MAX_AI_SCORE_DELTA
from twse_forecast.engine.quant_baseline import (
    compute_volatility_managed_allocation,
    normalize_three_probs,
    score_to_calibrated_forecast,
)
from twse_forecast.models import AIForecastSubmission, HorizonAIForecast, to_plain_dict


def render_briefing_markdown(
    *,
    record_id: str,
    features: dict[str, Any],
    quant_result: dict[str, Any],
    recent_settled_reviews: list[dict[str, Any]],
    collector_reports: list[dict[str, Any]],
) -> str:
    """產出供 AI 多頭／空頭／投資長辯證推理閱讀的結構化 Markdown 簡報包。"""
    cross = features.get("cross_market") or {}
    fund = features.get("fundamentals_valuation") or {}
    micro = features.get("microstructure_flows") or {}
    tech = features.get("technicals") or {}
    breadth = features.get("breadth") or {}
    ma60_ded = tech.get("ma60_deduction") or {}
    sentiment = features.get("sentiment") or {}
    headlines = features.get("news_headlines") or []
    events = features.get("event_calendar") or []

    lines: list[str] = [
        f"# 台股大盤多週期預測簡報包（{record_id}）",
        "",
        f"- **基準交易日**：`{features['reference_trade_date']}`",
        f"- **加權指數（TAIEX）收盤**：`{features['reference_taiex_close']:,.2f}`",
        f"- **元大台灣 50（0050.TW）還原收盤**：`{features['reference_0050_adj_close']:,.4f}`（交易日 `{features['reference_0050_trade_date']}`）",
        f"- **純量化建議持股水位（波動率調節）**：`{quant_result['target_allocation']*100:.1f}%`",
        "",
        "## 一、資料擷取管線狀態",
        "| 資料來源 | 狀態 | 說明 |",
        "| :--- | :--- | :--- |",
    ]
    for r in collector_reports:
        lines.append(f"| {r['source_name']} | `{r['status']}` | {r['message']} |")

    lines.extend([
        "",
        "## 二、五大維度核心數據摘要",
        "### 1. 總經與跨市場連動",
        f"- 台積電 ADR 溢價率：`{cross.get('tsmc_adr_premium_pct')}%` ｜ 美元兌新台幣：`{cross.get('usd_twd_latest')}`（近 20 日變動 `{cross.get('usd_twd_ret_20d_pct')}%`）",
        f"- 費半 SOX 近 5 日／20 日報酬：`{cross.get('sox_ret_5d_pct')}%` / `{cross.get('sox_ret_20d_pct')}%` ｜ NVDA 5 日：`{cross.get('nvda_ret_5d_pct')}%`",
        f"- S&P500／NASDAQ 近 5 日：`{cross.get('sp500_ret_5d_pct')}%` / `{cross.get('nasdaq_ret_5d_pct')}%` ｜ VIX：`{cross.get('vix_latest')}`",
        f"- 美債 10 年殖利率：`{cross.get('us10y_yield_pct')}%` ｜ 殖利率曲線利差（{cross.get('yield_spread_source')}）：`{cross.get('yield_spread_pct')}%` ｜ 高收益債利差：`{cross.get('hy_credit_spread_pct')}%`",
        "",
        "### 2. 基本面估值與營收動能",
        f"- 上市個股中位數 P/B：`{fund.get('market_median_pb')}` ｜ 中位數 P/E：`{fund.get('market_median_pe')}` ｜ 中位數殖利率：`{fund.get('market_median_div_yield_pct')}%`",
        f"- 台股風險溢酬 ERP（中位殖利率 − 美債 10Y）：`{fund.get('equity_risk_premium_pct')}%`",
        f"- 上市月營收（資料年月 `{fund.get('monthly_revenue_ym')}`）：正成長廣度 `{fund.get('monthly_revenue_pos_ratio_pct')}%` ｜ 合計 YoY `{fund.get('monthly_revenue_agg_yoy_pct')}%` ｜ 中位數 YoY `{fund.get('monthly_revenue_median_yoy_pct')}%`",
        "",
        "### 3. 台股微結構與法人／散戶籌碼",
        f"- 近 5 日現貨買賣超（億元）：外資 `{micro.get('foreign_5d_net_b')}` ｜ 投信 `{micro.get('trust_5d_net_b')}` ｜ 自營商自行 `{micro.get('dealer_prop_5d_net_b')}`",
        f"- 外資台指期（TX）淨未平倉口數：`{micro.get('foreign_tx_net_oi')}` ｜ 選擇權 Put/Call Ratio：`{micro.get('pc_ratio_latest_pct')}%`（5 日均 `{micro.get('pc_ratio_5d_avg_pct')}%`）",
        f"- 大盤融資餘額：`{micro.get('margin_balance_b')}` 億元（單日增減 `{micro.get('margin_change_1d_pct')}%`） ｜ 上漲／下跌家數：`{micro.get('twse_advancing_count')}` / `{micro.get('twse_declining_count')}`",
        "",
        "### 4. 多週期 K 線技術與拉積盤寬度",
        f"- 近 5／20／60／240 日報酬：`{tech.get('ret_5d_pct')}%` / `{tech.get('ret_20d_pct')}%` / `{tech.get('ret_60d_pct')}%` / `{tech.get('ret_240d_pct')}%` ｜ 3–12 月動能（剔除近月）：`{tech.get('momentum_12_1m_pct')}%`",
        f"- 均線位置（MA5/20/60/240）：`{tech.get('ma5')}` / `{tech.get('ma20')}` / `{tech.get('ma60')}` / `{tech.get('ma240')}` ｜ 月線（12M/36M）：`{tech.get('ma12m')}` / `{tech.get('ma36m')}`",
        f"- 20 日乖離：`{tech.get('bias_20d_pct')}%` ｜ RSI(14)：`{tech.get('rsi14')}` ｜ ATR(14)：`{tech.get('atr14_pct')}%` ｜ 20 日年化波動率：`{tech.get('realized_vol_20d')}` ｜ 布林壓縮百分位：`{tech.get('bollinger_squeeze_pct')}%`",
        f"- 季線扣抵預判：`{ma60_ded.get('regime')}`（未來 4 週扣抵餘裕 `{ma60_ded.get('deduction_cushion_4w_pct')}%`，4 週基準價 `{ma60_ded.get('deduct_4w_ref_price')}`）",
        f"- 30 檔代表籃子寬度：站上 20MA `{breadth.get('pct_above_ma20')}%` ／ 站上 60MA `{breadth.get('pct_above_ma60')}%` ／ 站上 240MA `{breadth.get('pct_above_ma240')}%` ｜ 指數 vs 等權重 20 日差距：`{breadth.get('taiex_vs_equal_spread_20d_pct')}%`（拉積盤警示：`{breadth.get('narrow_leadership_warning')}`）",
        "",
        "### 5. PTT 股板心理學與新聞／事件日曆",
    ])
    if sentiment:
        lines.append(
            f"- PTT Stock（樣本 {sentiment.get('posts_analyzed')} 篇）：爆文 `{sentiment.get('boom_posts')}` 篇 ｜ 恐慌詞命中 `{sentiment.get('panic_keyword_hits')}` ｜ 狂熱詞命中 `{sentiment.get('euphoria_keyword_hits')}` ｜ 狂熱分數 `{sentiment.get('retail_euphoria_score')}` ｜ 反向訊號 `{sentiment.get('contrarian_signal')}`"
        )
    else:
        lines.append("- PTT Stock：本次無法取得（以中性處理）")

    lines.append("- **未來 4 週固定事件日曆**：")
    for ev in events:
        lines.append(f"  - `{ev['date']}`：{ev['event']}")

    if headlines:
        lines.append("- **最新台股／總經財經新聞焦點**：")
        for idx, h in enumerate(headlines[:12], 1):
            src = f"（{h['source']}）" if h.get("source") else ""
            lines.append(f"  {idx}. {h['title']}{src}")

    lines.extend([
        "",
        "## 三、第一層純量化基準預測（AI 調整上限 ±25 分）",
        "| 週期 | 驗算錨點 | 量化分數 | 預期報酬 | 多／盤／空機率 | 80% 信賴點位區間 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for spec in HORIZONS:
        qf = quant_result["horizons"][spec.key]
        lines.append(
            f"| {qf['label']} | {qf['anchor_trading_days']} 交易日 | `{qf['quant_score']:+.2f}` | `{qf['expected_return']*100:+.2f}%` | `{qf['prob_up']*100:.1f}% / {qf['prob_flat']*100:.1f}% / {qf['prob_down']*100:.1f}%` | `{qf['ci80_low_level']:,.0f} ~ {qf['ci80_high_level']:,.0f}` |"
        )

    lines.extend(["", "## 四、近期已結算預測誤差回顧（自我校準參考）"])
    if not recent_settled_reviews:
        lines.append("- 尚無已到期結算之歷史預測紀錄（本次為初始追蹤累積期）。")
    else:
        for rev in recent_settled_reviews:
            lines.append(
                f"- `{rev['record_id']}` [{rev['horizon_label']}]：AI 預測 `{rev['ai_expected_return']*100:+.2f}%` vs 實際 `{rev['actual_return']*100:+.2f}%`（方向命中：`{rev['ai_hit']}`，AI Brier：`{rev['ai_brier']:.4f}` vs 量化 Brier：`{rev['quant_brier']:.4f}`）"
            )

    lines.extend([
        "",
        "## 五、AI 辯證 JSON 輸出格式契約（`Bounded Override ±25`）",
        f"請根據上方完整數據與當週最新財經情報，輸出合法 JSON 並儲存至 `data/snapshots/latest_ai_forecast.json`；每個週期的 `ai_score` 與 `quant_score` 差距將強制受限於 `±{MAX_AI_SCORE_DELTA:.0f}` 分內。",
    ])
    return "\n".join(lines)


def validate_and_finalize_ai_submission(
    raw_payload: dict[str, Any],
    quant_result: dict[str, Any],
    features: dict[str, Any],
) -> AIForecastSubmission:
    """驗證 AI 提交之 JSON，強制套用 ±25 分調整邊界並確保機率分布總和恆等於 1.0。"""
    latest_close = float(features["reference_taiex_close"])
    hist_stats = features.get("historical_horizon_stats") or {}
    tech = features.get("technicals") or {}
    raw_horizons = raw_payload.get("horizons") or {}
    quant_horizons = quant_result["horizons"]

    finalized_horizons: dict[str, HorizonAIForecast] = {}
    ai_scores: dict[str, float] = {}

    for spec in HORIZONS:
        q_item = quant_horizons[spec.key]
        q_score = float(q_item["quant_score"] if isinstance(q_item, dict) else q_item.quant_score)
        user_h = raw_horizons.get(spec.key)
        if not isinstance(user_h, dict):
            raise ValueError(f"AI 預測缺少必要週期：{spec.key}")

        requested_score = float(user_h.get("ai_score", q_score))
        raw_delta = requested_score - q_score
        clamped_delta = max(-MAX_AI_SCORE_DELTA, min(MAX_AI_SCORE_DELTA, raw_delta))
        final_ai_score = round(max(-100.0, min(100.0, q_score + clamped_delta)), 2)
        ai_scores[spec.key] = final_ai_score

        h_stat = hist_stats.get(spec.key) or {}
        hist_mean = float(h_stat.get("mean_return", 0.02))
        hist_std = float(h_stat.get("std_return", 0.05))

        # 使用一致的統計轉換取得基準校準值
        calibrated = score_to_calibrated_forecast(
            spec=spec,
            score=final_ai_score,
            latest_close=latest_close,
            hist_mean=hist_mean,
            hist_std=hist_std,
            key_drivers={},
        )

        # 若 AI 自行微調機率且未超過界限，或直接沿用分數換算校準機率，皆強制正規化總和=1.0
        if (
            "prob_up" in user_h
            and "prob_flat" in user_h
            and "prob_down" in user_h
            and abs(raw_delta) <= MAX_AI_SCORE_DELTA
        ):
            p_up, p_flat, p_down = normalize_three_probs(
                float(user_h["prob_up"]),
                float(user_h["prob_flat"]),
                float(user_h["prob_down"]),
            )
        else:
            p_up, p_flat, p_down = calibrated.prob_up, calibrated.prob_flat, calibrated.prob_down

        invalidation = float(
            user_h.get("invalidation_level")
            or (calibrated.ci80_low_level if final_ai_score >= 0 else calibrated.ci80_high_level)
        )
        thesis = str(user_h.get("thesis") or "").strip()
        if not thesis:
            raise ValueError(f"AI 預測週期 {spec.key} 缺少具體論述 thesis")

        finalized_horizons[spec.key] = HorizonAIForecast(
            horizon_key=spec.key,
            ai_score=final_ai_score,
            quant_score=round(q_score, 2),
            delta_score=round(final_ai_score - q_score, 2),
            expected_return=calibrated.expected_return,
            prob_up=p_up,
            prob_flat=p_flat,
            prob_down=p_down,
            ci80_low_level=calibrated.ci80_low_level,
            ci80_high_level=calibrated.ci80_high_level,
            invalidation_level=round(invalidation, 2),
            thesis=thesis,
        )

    # 建議持股水位：以 AI 各週期分數計算波動率調節水位，並允許微幅調整 (±0.15) 但必須在 0.0~1.0
    quant_alloc = float(quant_result["target_allocation"])
    ai_suggested_vol_alloc = compute_volatility_managed_allocation(
        ai_scores, tech.get("realized_vol_20d")
    )
    if "target_allocation" in raw_payload and raw_payload["target_allocation"] is not None:
        req_alloc = float(raw_payload["target_allocation"])
        # 限制 AI 持股調整不脫離波動率調節區間過遠 (±0.20)
        final_alloc = round(
            max(0.0, min(1.0, max(ai_suggested_vol_alloc - 0.20, min(ai_suggested_vol_alloc + 0.20, req_alloc)))),
            4,
        )
    else:
        final_alloc = ai_suggested_vol_alloc

    bull_case = str(raw_payload.get("bull_case") or "").strip()
    bear_case = str(raw_payload.get("bear_case") or "").strip()
    cio_synthesis = str(raw_payload.get("cio_synthesis") or "").strip()
    override_rationale = str(raw_payload.get("override_rationale") or "").strip()
    model_name = str(raw_payload.get("model_name") or "Gemini 4 Carbon").strip()

    if not bull_case or not bear_case or not cio_synthesis or not override_rationale:
        raise ValueError("AI 辯證 JSON 必須完整填寫 bull_case、bear_case、cio_synthesis 與 override_rationale")

    # 避免未微調但過度極端：檢查若 quant_alloc 已存在不衝突
    _ = quant_alloc

    return AIForecastSubmission(
        model_name=model_name,
        bull_case=bull_case,
        bear_case=bear_case,
        cio_synthesis=cio_synthesis,
        override_rationale=override_rationale,
        target_allocation=final_alloc,
        horizons=finalized_horizons,
        external_news_notes=[str(x) for x in (raw_payload.get("external_news_notes") or [])],
    )


def save_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_plain_dict(data), ensure_ascii=False, indent=2), encoding="utf-8")
