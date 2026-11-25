"""閉環統計自我校準與偏差診斷模組：經驗波動率校準、系統性偏差偵測與動態 AI 授權韁繩。"""

from __future__ import annotations

from typing import Any

from twse_forecast.config import HORIZONS, MAX_AI_SCORE_DELTA

# 至少累積 6 筆已到期結算樣本後才啟動參數微調，防止小樣本過度擬合
MIN_CALIBRATION_SAMPLES = 6


def compute_calibration_profile(
    records: list[dict[str, Any]],
    settlements: list[dict[str, Any]],
) -> dict[str, Any]:
    """統計各週期歷史預測偏差、區間涵蓋率與 AI 淨貢獻度，計算有界校準參數。"""
    rec_map = {r["record_id"]: r for r in records}
    by_horizon: dict[str, dict[str, Any]] = {}
    high_conf_misses: list[dict[str, Any]] = []

    for s in settlements:
        rec = rec_map.get(s["record_id"])
        max_prob = 0.0
        if rec:
            af = ((rec.get("ai_forecast") or {}).get("horizons") or {}).get(s["horizon_key"]) or {}
            max_prob = max(
                float(af.get("prob_up", 0.0)),
                float(af.get("prob_flat", 0.0)),
                float(af.get("prob_down", 0.0)),
            )
        if not s.get("ai_hit") and max_prob >= 0.58:
            high_conf_misses.append({
                "record_id": s["record_id"],
                "horizon_key": s["horizon_key"],
                "horizon_label": s["horizon_label"],
                "predicted_class": s.get("ai_pred_class", "?"),
                "max_prob": round(max_prob, 4),
                "actual_outcome": s["actual_outcome"],
                "actual_return": float(s["actual_return"]),
            })

    for spec in HORIZONS:
        sub = [s for s in settlements if s["horizon_key"] == spec.key]
        n = len(sub)
        if n == 0:
            by_horizon[spec.key] = {
                "horizon_label": spec.label,
                "settled_count": 0,
                "active": False,
                "mean_bias": 0.0,
                "empirical_ci80_cov": None,
                "ai_brier_value_add": 0.0,
                "override_win_rate": None,
                "vol_multiplier": 1.0,
                "max_ai_delta": MAX_AI_SCORE_DELTA,
                "bias_direction": "neutral",
                "status_note": "尚無到期樣本（維持預設基準參數）",
            }
            continue

        mean_bias = sum(float(x["ai_expected_return"]) - float(x["actual_return"]) for x in sub) / n
        ci80_cov = sum(1 for x in sub if x.get("in_ai_ci80")) / n
        brier_va = sum(float(x["quant_brier"]) - float(x["ai_brier"]) for x in sub) / n

        override_wins = 0
        override_total = 0
        for x in sub:
            rec = rec_map.get(x["record_id"])
            if not rec:
                continue
            af = ((rec.get("ai_forecast") or {}).get("horizons") or {}).get(spec.key) or {}
            if abs(float(af.get("delta_score", 0.0))) >= 3.0:
                override_total += 1
                if float(x["ai_brier"]) < float(x["quant_brier"]):
                    override_wins += 1
        override_win_rate = round(override_wins / override_total, 4) if override_total > 0 else None

        if mean_bias > 0.02:
            bias_dir = "over_optimistic"
        elif mean_bias < -0.02:
            bias_dir = "over_pessimistic"
        else:
            bias_dir = "balanced"

        if n < MIN_CALIBRATION_SAMPLES:
            vol_mult = 1.0
            max_delta = MAX_AI_SCORE_DELTA
            note = f"樣本累積中 ({n}/{MIN_CALIBRATION_SAMPLES})，供 AI 推理反思，暫不縮放硬參數"
            active = False
        else:
            active = True
            # 1. 波動率區間自校準（有界區間 0.85 ~ 1.35）
            if ci80_cov < 0.78:
                vol_mult = round(min(1.35, 1.0 + (0.80 - ci80_cov) * 0.90), 4)
            elif ci80_cov > 0.92:
                vol_mult = round(max(0.85, 1.0 - (ci80_cov - 0.85) * 0.60), 4)
            else:
                vol_mult = 1.0

            # 2. 動態 AI 授權韁繩（若主觀調整持續劣於純量化，自動收緊調整幅度上限）
            if brier_va < -0.04:
                max_delta = 12.0
            elif brier_va < -0.015:
                max_delta = 16.0
            else:
                max_delta = MAX_AI_SCORE_DELTA

            parts = []
            if bias_dir == "over_optimistic":
                parts.append(f"系統性偏多高估 (+{mean_bias*100:.1f}%)，請適度收斂樂觀預期")
            elif bias_dir == "over_pessimistic":
                parts.append(f"系統性偏空低估 ({mean_bias*100:.1f}%)，請避免過度恐慌")
            else:
                parts.append("報酬均值偏差均衡")
            if max_delta < MAX_AI_SCORE_DELTA:
                parts.append(f"近期主觀調整落後量化，授權收緊至 ±{max_delta:.0f} 分")
            if vol_mult > 1.02:
                parts.append(f"區間涵蓋偏低，波動率放大 {vol_mult:.2f}x")
            note = "；".join(parts)

        by_horizon[spec.key] = {
            "horizon_label": spec.label,
            "settled_count": n,
            "active": active,
            "mean_bias": round(mean_bias, 5),
            "empirical_ci80_cov": round(ci80_cov, 4),
            "ai_brier_value_add": round(brier_va, 4),
            "override_win_rate": override_win_rate,
            "vol_multiplier": vol_mult,
            "max_ai_delta": max_delta,
            "bias_direction": bias_dir,
            "status_note": note,
        }

    return {
        "min_samples_threshold": MIN_CALIBRATION_SAMPLES,
        "total_settled": len(settlements),
        "by_horizon": by_horizon,
        "high_confidence_misses": high_conf_misses[-3:],
    }


def render_calibration_diagnostics_markdown(cal_profile: dict[str, Any] | None) -> list[str]:
    """將閉環自我校準與滾動偏差診斷轉為簡報包 Markdown 段落。"""
    if not cal_profile or cal_profile.get("total_settled", 0) == 0:
        return [
            "- **閉環自我校準狀態**：目前尚無已到期結算樣本（各週期維持預設波動率乘數 `1.00x` 與完整 `±25` 分授權上限；滿 6 筆結算後自動啟動經驗波動率校準與動態授權韁繩）。"
        ]

    lines = [
        "### 滾動偏差與自我校準診斷表",
        "| 週期 | 已結算數 | 平均預測偏差 (AI−實際) | 80% 區間實際涵蓋 | AI 淨精度改善 | 波動校準乘數 | 本週允許調整上限 | 校準指引 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    by_h = cal_profile.get("by_horizon") or {}
    for spec in HORIZONS:
        h = by_h.get(spec.key) or {}
        if h.get("settled_count", 0) == 0:
            lines.append(
                f"| {spec.label} | 0 | - | - | - | `1.00x` | `±{MAX_AI_SCORE_DELTA:.0f}` | 尚未到期 |"
            )
            continue
        cov_txt = f"{h['empirical_ci80_cov']*100:.1f}%" if h.get("empirical_ci80_cov") is not None else "-"
        lines.append(
            f"| {spec.label} | {h['settled_count']} | `{h['mean_bias']*100:+.2f}%` | {cov_txt} | "
            f"`{h['ai_brier_value_add']:+.4f}` | `{h['vol_multiplier']:.2f}x` | "
            f"**`±{h['max_ai_delta']:.0f}`** | {h['status_note']} |"
        )

    misses = cal_profile.get("high_confidence_misses") or []
    if misses:
        lines.append("")
        lines.append("### 近期高信心誤判警示（推理時務必檢核避免重蹈覆轍）")
        for m in misses:
            lines.append(
                f"- `{m['record_id']}` [{m['horizon_label']}]：當時預測 `{m['predicted_class']}`（信心 `{m['max_prob']*100:.1f}%`），實際結果為 `{m['actual_outcome']}`（報酬 `{m['actual_return']*100:+.2f}%`）。"
            )
    return lines
