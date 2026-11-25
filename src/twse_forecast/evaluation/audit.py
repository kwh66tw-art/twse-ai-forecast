"""到期預測結算、統計驗證（Brier Score／Hit Rate／Coverage）、未到期軌跡追蹤與模擬部位績效引擎。"""

from __future__ import annotations

import math
from typing import Any

from twse_forecast.collectors.http_utils import now_taipei_iso
from twse_forecast.config import ETF_FEE_RATE, ETF_SELL_TAX_RATE, HORIZON_MAP, HORIZONS
from twse_forecast.models import MarketSeries


def classify_return(actual_ret: float, flat_threshold: float) -> str:
    if actual_ret > flat_threshold:
        return "up"
    if actual_ret < -flat_threshold:
        return "down"
    return "flat"


def compute_brier_3class(p_up: float, p_flat: float, p_down: float, actual_outcome: str) -> float:
    o_up = 1.0 if actual_outcome == "up" else 0.0
    o_flat = 1.0 if actual_outcome == "flat" else 0.0
    o_down = 1.0 if actual_outcome == "down" else 0.0
    return round(
        (p_up - o_up) ** 2 + (p_flat - o_flat) ** 2 + (p_down - o_down) ** 2,
        6,
    )


def _predicted_class(p_up: float, p_flat: float, p_down: float) -> str:
    pairs = [("up", p_up), ("flat", p_flat), ("down", p_down)]
    return max(pairs, key=lambda x: x[1])[0]


def _find_anchor_bar_index(dates: list[str], reference_trade_date: str) -> int | None:
    """找出對應參考基準交易日（或之後最近交易日）的索引。"""
    for idx, d in enumerate(dates):
        if d >= reference_trade_date:
            return idx
    return None


def settle_matured_predictions(
    records: list[dict[str, Any]],
    taiex_series: MarketSeries,
    existing_settlement_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """比對歷史鎖定預測與最新 ^TWII K 線，結算所有已屆滿錨點交易日的預測。"""
    dates = taiex_series.dates()
    bars = taiex_series.bars
    settled_at = now_taipei_iso()
    already_settled = existing_settlement_ids or set()
    newly_settled: list[dict[str, Any]] = []

    for rec in records:
        ref_date = rec["reference_trade_date"]
        ref_close = float(rec["reference_taiex_close"])
        start_idx = _find_anchor_bar_index(dates, ref_date)
        if start_idx is None or ref_close <= 0:
            continue

        quant_h = rec["quant_baseline"]["horizons"]
        ai_h = rec["ai_forecast"]["horizons"]

        for spec in HORIZONS:
            s_id = f"{rec['record_id']}:{spec.key}"
            if s_id in already_settled:
                continue
            target_idx = start_idx + spec.anchor_trading_days
            if target_idx >= len(bars):
                continue  # 尚未到期

            target_bar = bars[target_idx]
            actual_ret = (target_bar.close / ref_close) - 1.0
            actual_outcome = classify_return(actual_ret, spec.flat_threshold)

            qf = quant_h[spec.key]
            af = ai_h[spec.key]

            q_pred = _predicted_class(qf["prob_up"], qf["prob_flat"], qf["prob_down"])
            a_pred = _predicted_class(af["prob_up"], af["prob_flat"], af["prob_down"])

            q_brier = compute_brier_3class(qf["prob_up"], qf["prob_flat"], qf["prob_down"], actual_outcome)
            a_brier = compute_brier_3class(af["prob_up"], af["prob_flat"], af["prob_down"], actual_outcome)

            # 無腦永遠看多基準（Naive Bull: 60% up / 25% flat / 15% down）
            naive_bull_brier = compute_brier_3class(0.60, 0.25, 0.15, actual_outcome)

            newly_settled.append({
                "settlement_id": s_id,
                "record_id": rec["record_id"],
                "horizon_key": spec.key,
                "horizon_label": spec.label,
                "settled_at_iso": settled_at,
                "reference_trade_date": ref_date,
                "reference_taiex_close": ref_close,
                "target_trade_date": target_bar.date,
                "target_taiex_close": round(target_bar.close, 2),
                "actual_return": round(actual_ret, 5),
                "actual_outcome": actual_outcome,
                "ai_expected_return": float(af["expected_return"]),
                "quant_expected_return": float(qf["expected_return"]),
                "ai_pred_class": a_pred,
                "quant_pred_class": q_pred,
                "ai_hit": a_pred == actual_outcome,
                "quant_hit": q_pred == actual_outcome,
                "naive_bull_hit": actual_outcome == "up",
                "ai_brier": a_brier,
                "quant_brier": q_brier,
                "naive_bull_brier": naive_bull_brier,
                "in_ai_ci80": float(af["ci80_low_level"]) <= target_bar.close <= float(af["ci80_high_level"]),
                "in_quant_ci80": float(qf["ci80_low_level"]) <= target_bar.close <= float(qf["ci80_high_level"]),
            })

    return newly_settled


def compute_open_trajectory_status(
    records: list[dict[str, Any]],
    taiex_series: MarketSeries,
) -> list[dict[str, Any]]:
    """針對尚未到期的中長週期（特別是 6–12 個月、1–3 年）計算階段性軌跡通道一致性。"""
    if not records or not taiex_series.bars:
        return []
    dates = taiex_series.dates()
    bars = taiex_series.bars
    latest_bar = bars[-1]
    out: list[dict[str, Any]] = []

    for rec in records:
        ref_date = rec["reference_trade_date"]
        ref_close = float(rec["reference_taiex_close"])
        start_idx = _find_anchor_bar_index(dates, ref_date)
        if start_idx is None or ref_close <= 0:
            continue
        elapsed_days = (len(bars) - 1) - start_idx
        if elapsed_days < 5:
            continue

        segment_closes = [b.close for b in bars[start_idx:]]
        mfe_pct = ((max(segment_closes) / ref_close) - 1.0) * 100.0
        mae_pct = ((min(segment_closes) / ref_close) - 1.0) * 100.0
        curr_ret_pct = ((latest_bar.close / ref_close) - 1.0) * 100.0

        for spec in HORIZONS:
            if elapsed_days >= spec.anchor_trading_days:
                continue
            af = rec["ai_forecast"]["horizons"][spec.key]
            qf = rec["quant_baseline"]["horizons"][spec.key]
            t_ratio = elapsed_days / spec.anchor_trading_days
            std_dev = float(qf["std_dev"])
            scaled_exp = float(af["expected_return"]) * t_ratio
            scaled_half_band = 1.28155 * std_dev * math.sqrt(t_ratio)

            corridor_low = round(ref_close * (1.0 + scaled_exp - scaled_half_band), 2)
            corridor_high = round(ref_close * (1.0 + scaled_exp + scaled_half_band), 2)
            in_corridor = corridor_low <= latest_bar.close <= corridor_high

            out.append({
                "record_id": rec["record_id"],
                "horizon_key": spec.key,
                "horizon_label": spec.label,
                "elapsed_trading_days": elapsed_days,
                "anchor_trading_days": spec.anchor_trading_days,
                "progress_pct": round(t_ratio * 100.0, 1),
                "current_return_pct": round(curr_ret_pct, 2),
                "corridor_low": corridor_low,
                "corridor_high": corridor_high,
                "in_scaled_ci80_corridor": in_corridor,
                "mfe_pct": round(mfe_pct, 2),
                "mae_pct": round(mae_pct, 2),
            })

    return out


def _max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        if peak > 0:
            dd = (val / peak) - 1.0
            if dd < max_dd:
                max_dd = dd
    return max_dd


def _annualized_sharpe(period_returns: list[float], periods_per_year: float = 52.0, rf_annual: float = 0.015) -> float:
    if len(period_returns) < 2:
        return 0.0
    rf_per = rf_annual / periods_per_year
    excess = [r - rf_per for r in period_returns]
    mean_ex = sum(excess) / len(excess)
    var = sum((x - mean_ex) ** 2 for x in excess) / (len(excess) - 1)
    std = math.sqrt(var)
    if std <= 1e-9:
        return 0.0
    return (mean_ex / std) * math.sqrt(periods_per_year)


def compute_shadow_portfolio(
    records: list[dict[str, Any]],
    etf0050_series: MarketSeries,
    initial_capital: float = 1_000_000.0,
) -> dict[str, Any]:
    """以 0050.TW 還原權息收盤價計算四種策略之模擬投資組合實戰績效（含真實交易成本）。"""
    if not records or not etf0050_series.bars:
        return {"available": False, "reason": "尚無預測紀錄或 0050 資料"}

    latest_0050 = float(etf0050_series.bars[-1].adj_close)
    latest_date = etf0050_series.bars[-1].date

    # 建構評價時點清單：每筆預測的進場點 + 若最新交易日較最後一筆更新則補上期末市值結算點
    checkpoints: list[dict[str, Any]] = []
    for r in records:
        checkpoints.append({
            "label": r["record_id"],
            "date": r["reference_trade_date"],
            "price": float(r["reference_0050_adj_close"]),
            "ai_alloc": float(r["ai_forecast"]["target_allocation"]),
            "quant_alloc": float(r["quant_baseline"]["target_allocation"]),
        })

    if latest_date > checkpoints[-1]["date"] and latest_0050 > 0:
        checkpoints.append({
            "label": f"M2M({latest_date})",
            "date": latest_date,
            "price": latest_0050,
            "ai_alloc": checkpoints[-1]["ai_alloc"],
            "quant_alloc": checkpoints[-1]["quant_alloc"],
        })

    if len(checkpoints) < 2:
        return {
            "available": True,
            "periods": 0,
            "initial_capital": initial_capital,
            "note": "目前僅有基期建倉點，將於下週起產生第一段實戰報酬曲線。",
            "strategies": {
                "ai": {"equity": initial_capital, "total_return_pct": 0.0, "sharpe": 0.0, "max_drawdown_pct": 0.0, "total_cost": 0.0},
                "quant": {"equity": initial_capital, "total_return_pct": 0.0, "sharpe": 0.0, "max_drawdown_pct": 0.0, "total_cost": 0.0},
                "buy_hold_0050": {"equity": initial_capital, "total_return_pct": 0.0, "sharpe": 0.0, "max_drawdown_pct": 0.0, "total_cost": 0.0},
                "balanced_60_40": {"equity": initial_capital, "total_return_pct": 0.0, "sharpe": 0.0, "max_drawdown_pct": 0.0, "total_cost": 0.0},
            },
            "curve": [
                {
                    "label": checkpoints[0]["label"],
                    "date": checkpoints[0]["date"],
                    "ai_equity": initial_capital,
                    "quant_equity": initial_capital,
                    "buy_hold_equity": initial_capital,
                    "balanced_60_40_equity": initial_capital,
                    "ai_alloc_pct": round(checkpoints[0]["ai_alloc"] * 100.0, 1),
                }
            ],
        }

    def simulate(allocs: list[float], is_buy_hold: bool = False) -> dict[str, Any]:
        # 初期建倉扣一次買入手續費
        w_curr = 1.0 if is_buy_hold else allocs[0]
        init_fee = initial_capital * w_curr * ETF_FEE_RATE
        equity = initial_capital - init_fee
        total_cost = init_fee
        curve = [round(equity, 2)]
        period_rets: list[float] = []
        rf_weekly = 0.015 / 52.0

        for i in range(1, len(checkpoints)):
            p0 = checkpoints[i - 1]["price"]
            p1 = checkpoints[i]["price"]
            etf_ret = (p1 / p0 - 1.0) if p0 > 0 else 0.0

            gross_ret = w_curr * etf_ret + (1.0 - w_curr) * rf_weekly
            prev_eq = equity
            equity = equity * (1.0 + gross_ret)

            if not is_buy_hold and i < len(checkpoints) - 1:
                w_next = allocs[i]
                # 考量資產價格變動後的實際漂移比重
                current_etf_value = prev_eq * w_curr * (1.0 + etf_ret)
                actual_w = current_etf_value / equity if equity > 0 else 0.0
                delta_w = w_next - actual_w
                turnover_amt = abs(delta_w) * equity
                if delta_w > 0:
                    cost = turnover_amt * ETF_FEE_RATE
                elif delta_w < 0:
                    cost = turnover_amt * (ETF_FEE_RATE + ETF_SELL_TAX_RATE)
                else:
                    cost = 0.0
                equity -= cost
                total_cost += cost
                w_curr = w_next

            period_rets.append((equity / prev_eq) - 1.0 if prev_eq > 0 else 0.0)
            curve.append(round(equity, 2))

        return {
            "equity": round(equity, 2),
            "total_return_pct": round((equity / initial_capital - 1.0) * 100.0, 3),
            "sharpe": round(_annualized_sharpe(period_rets), 3),
            "max_drawdown_pct": round(_max_drawdown(curve) * 100.0, 3),
            "total_cost": round(total_cost, 2),
            "curve": curve,
        }

    ai_sim = simulate([c["ai_alloc"] for c in checkpoints])
    quant_sim = simulate([c["quant_alloc"] for c in checkpoints])
    bh_sim = simulate([1.0 for _ in checkpoints], is_buy_hold=True)
    bal_sim = simulate([0.60 for _ in checkpoints])

    curve_rows = []
    for idx, cp in enumerate(checkpoints):
        curve_rows.append({
            "label": cp["label"],
            "date": cp["date"],
            "ai_equity": ai_sim["curve"][idx],
            "quant_equity": quant_sim["curve"][idx],
            "buy_hold_equity": bh_sim["curve"][idx],
            "balanced_60_40_equity": bal_sim["curve"][idx],
            "ai_alloc_pct": round(cp["ai_alloc"] * 100.0, 1),
        })

    for d in (ai_sim, quant_sim, bh_sim, bal_sim):
        d.pop("curve", None)

    return {
        "available": True,
        "periods": len(checkpoints) - 1,
        "initial_capital": initial_capital,
        "strategies": {
            "ai": ai_sim,
            "quant": quant_sim,
            "buy_hold_0050": bh_sim,
            "balanced_60_40": bal_sim,
        },
        "ai_alpha_vs_buy_hold_pct": round(
            ai_sim["total_return_pct"] - bh_sim["total_return_pct"], 3
        ),
        "ai_value_add_vs_quant_pct": round(
            ai_sim["total_return_pct"] - quant_sim["total_return_pct"], 3
        ),
        "curve": curve_rows,
    }


def summarize_monthly_audit(
    records: list[dict[str, Any]],
    settlements: list[dict[str, Any]],
    taiex_series: MarketSeries,
    etf0050_series: MarketSeries,
) -> dict[str, Any]:
    """產出完整的月度／累計績效稽核統計摘要。"""
    by_horizon: dict[str, dict[str, Any]] = {}
    for spec in HORIZONS:
        sub = [s for s in settlements if s["horizon_key"] == spec.key]
        if not sub:
            by_horizon[spec.key] = {
                "horizon_label": spec.label,
                "settled_count": 0,
            }
            continue
        n = len(sub)
        ai_hits = sum(1 for x in sub if x["ai_hit"])
        q_hits = sum(1 for x in sub if x["quant_hit"])
        base_up = sum(1 for x in sub if x["actual_outcome"] == "up")
        ai_brier_avg = sum(x["ai_brier"] for x in sub) / n
        q_brier_avg = sum(x["quant_brier"] for x in sub) / n
        naive_brier_avg = sum(x["naive_bull_brier"] for x in sub) / n
        ai_ci_cov = sum(1 for x in sub if x["in_ai_ci80"]) / n
        q_ci_cov = sum(1 for x in sub if x["in_quant_ci80"]) / n
        ai_mae = sum(abs(x["ai_expected_return"] - x["actual_return"]) for x in sub) / n

        by_horizon[spec.key] = {
            "horizon_label": spec.label,
            "settled_count": n,
            "ai_hit_rate_pct": round(ai_hits / n * 100.0, 2),
            "quant_hit_rate_pct": round(q_hits / n * 100.0, 2),
            "market_up_base_rate_pct": round(base_up / n * 100.0, 2),
            "ai_brier_score": round(ai_brier_avg, 4),
            "quant_brier_score": round(q_brier_avg, 4),
            "naive_bull_brier_score": round(naive_brier_avg, 4),
            "ai_brier_value_add": round(q_brier_avg - ai_brier_avg, 4),  # 正值代表 AI 比純量化更準
            "ai_ci80_coverage_pct": round(ai_ci_cov * 100.0, 1),
            "quant_ci80_coverage_pct": round(q_ci_cov * 100.0, 1),
            "ai_mae_pct": round(ai_mae * 100.0, 2),
        }

    open_trajectories = compute_open_trajectory_status(records, taiex_series)
    portfolio = compute_shadow_portfolio(records, etf0050_series)

    return {
        "generated_at": now_taipei_iso(),
        "total_weekly_records": len(records),
        "total_settled_horizons": len(settlements),
        "by_horizon": by_horizon,
        "open_trajectories": open_trajectories,
        "shadow_portfolio": portfolio,
    }
