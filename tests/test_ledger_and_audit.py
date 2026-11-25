"""防竄改 SHA-256 帳本、排程槽位標記、到期結算與扣費模擬組合測試。"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from twse_forecast.config import HORIZONS, TAIPEI_TZ, ProjectPaths
from twse_forecast.evaluation.audit import (
    compute_brier_3class,
    compute_open_trajectory_status,
    compute_shadow_portfolio,
    settle_matured_predictions,
)
from twse_forecast.ledger.store import (
    append_prediction_record,
    compute_schedule_metadata,
    load_ledger_records,
    verify_ledger,
)
from twse_forecast.models import MarketSeries, OHLCVBar


def _fake_quant_and_ai(alloc: float, p_up: float = 0.6, p_flat: float = 0.25, p_down: float = 0.15) -> tuple[dict, dict]:
    q_h = {}
    ai_h = {}
    for spec in HORIZONS:
        q_h[spec.key] = {
            "horizon_key": spec.key,
            "label": spec.label,
            "anchor_trading_days": spec.anchor_trading_days,
            "quant_score": 20.0,
            "expected_return": 0.03,
            "std_dev": 0.05,
            "prob_up": 0.50,
            "prob_flat": 0.30,
            "prob_down": 0.20,
            "ci80_low_return": -0.03,
            "ci80_high_return": 0.09,
            "ci80_low_level": 19400.0,
            "ci80_high_level": 21800.0,
            "key_drivers": {},
        }
        ai_h[spec.key] = {
            "horizon_key": spec.key,
            "ai_score": 30.0,
            "quant_score": 20.0,
            "delta_score": 10.0,
            "expected_return": 0.04,
            "prob_up": p_up,
            "prob_flat": p_flat,
            "prob_down": p_down,
            "ci80_low_level": 19500.0,
            "ci80_high_level": 22000.0,
            "invalidation_level": 19500.0,
            "thesis": "測試論述",
        }
    return (
        {"target_allocation": alloc, "horizons": q_h},
        {
            "model_name": "Gemini",
            "bull_case": "B",
            "bear_case": "R",
            "cio_synthesis": "S",
            "override_rationale": "O",
            "target_allocation": alloc,
            "horizons": ai_h,
        },
    )


def test_schedule_metadata_late_flag() -> None:
    monday_8am = datetime(2026, 11, 23, 8, 5, 0, tzinfo=TAIPEI_TZ)
    meta_on_time = compute_schedule_metadata(monday_8am)
    assert meta_on_time["is_late_catchup"] is False

    tuesday_night = datetime(2026, 11, 24, 21, 0, 0, tzinfo=TAIPEI_TZ)
    meta_late = compute_schedule_metadata(tuesday_night)
    assert meta_late["is_late_catchup"] is True


def test_ledger_hash_chain_and_tamper_detection(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    q1, a1 = _fake_quant_and_ai(0.7)
    q2, a2 = _fake_quant_and_ai(0.5)

    append_prediction_record(
        paths,
        schedule_meta={"record_id": "2026-W47", "scheduled_slot": "2026-11-16T08:00:00+08:00", "executed_at_iso": "2026-11-16T08:01:00+08:00", "is_late_catchup": False},
        features={"reference_trade_date": "2026-11-13", "reference_taiex_close": 20000.0, "reference_0050_adj_close": 180.0},
        quant_baseline=q1,
        ai_forecast=a1,
        snapshot_path="data/snapshots/2026-W47_snapshot.json",
    )
    append_prediction_record(
        paths,
        schedule_meta={"record_id": "2026-W48", "scheduled_slot": "2026-11-23T08:00:00+08:00", "executed_at_iso": "2026-11-23T08:01:00+08:00", "is_late_catchup": False},
        features={"reference_trade_date": "2026-11-20", "reference_taiex_close": 20500.0, "reference_0050_adj_close": 185.0},
        quant_baseline=q2,
        ai_forecast=a2,
        snapshot_path="data/snapshots/2026-W48_snapshot.json",
    )

    ok, issues, total = verify_ledger(paths)
    assert ok is True
    assert issues == []
    assert total == 2

    # 拒絕同週覆寫
    with pytest.raises(ValueError):
        append_prediction_record(
            paths,
            schedule_meta={"record_id": "2026-W48", "scheduled_slot": "2026-11-23T08:00:00+08:00", "executed_at_iso": "2026-11-23T09:30:00+08:00", "is_late_catchup": True},
            features={"reference_trade_date": "2026-11-20", "reference_taiex_close": 20500.0, "reference_0050_adj_close": 185.0},
            quant_baseline=q2,
            ai_forecast=a2,
            snapshot_path="data/snapshots/2026-W48_snapshot.json",
        )

    # 刻意竄改 JSONL 內容中的數字，驗證 SHA-256 立即捕捉
    raw = paths.ledger_jsonl.read_text(encoding="utf-8")
    tampered = raw.replace("20000.0", "29999.0")
    paths.ledger_jsonl.write_text(tampered, encoding="utf-8")
    ok_after, issues_after, _ = verify_ledger(paths)
    assert ok_after is False
    assert any("竄改" in msg for msg in issues_after)


def test_settlement_brier_trajectory_and_shadow_portfolio(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    q1, a1 = _fake_quant_and_ai(0.8, p_up=0.7, p_flat=0.2, p_down=0.1)
    q2, a2 = _fake_quant_and_ai(0.4, p_up=0.4, p_flat=0.4, p_down=0.2)

    append_prediction_record(
        paths,
        schedule_meta={"record_id": "2026-W40", "scheduled_slot": "2026-10-05T08:00:00+08:00", "executed_at_iso": "2026-10-05T08:01:00+08:00", "is_late_catchup": False},
        features={"reference_trade_date": "2026-10-02", "reference_taiex_close": 20000.0, "reference_0050_adj_close": 180.0},
        quant_baseline=q1,
        ai_forecast=a1,
        snapshot_path="s1.json",
    )
    append_prediction_record(
        paths,
        schedule_meta={"record_id": "2026-W45", "scheduled_slot": "2026-11-09T08:00:00+08:00", "executed_at_iso": "2026-11-09T08:01:00+08:00", "is_late_catchup": False},
        features={"reference_trade_date": "2026-11-06", "reference_taiex_close": 21000.0, "reference_0050_adj_close": 190.0},
        quant_baseline=q2,
        ai_forecast=a2,
        snapshot_path="s2.json",
    )

    # 造出 40 天交易日序列（2026-10-02 起算滿 20 個交易日後應結算 W40 的 1_4w，未滿 40 日不結算 1_3m）
    start = date(2026, 10, 2)
    bars_twii = []
    bars_0050 = []
    trading_day = 0
    for offset in range(60):
        d = start + timedelta(days=offset)
        if d.weekday() >= 5:
            continue
        price = 20000.0 + trading_day * 50.0  # 第 20 個交易日 = 21000 (+5%) -> up
        etf_p = 180.0 + trading_day * 0.5
        bars_twii.append(OHLCVBar(date=d.isoformat(), open=price, high=price, low=price, close=price, adj_close=price, volume=100.0))
        bars_0050.append(OHLCVBar(date=d.isoformat(), open=etf_p, high=etf_p, low=etf_p, close=etf_p, adj_close=etf_p, volume=100.0))
        trading_day += 1
        if trading_day >= 35:
            break

    taiex_ser = MarketSeries(symbol="^TWII", bars=bars_twii)
    etf_ser = MarketSeries(symbol="0050.TW", bars=bars_0050)
    records = load_ledger_records(paths)

    settled = settle_matured_predictions(records, taiex_ser)
    # 僅 W40 的 1_4w (anchor=20) 到期
    assert len(settled) == 1
    assert settled[0]["record_id"] == "2026-W40"
    assert settled[0]["horizon_key"] == "1_4w"
    assert settled[0]["actual_outcome"] == "up"
    assert settled[0]["ai_hit"] is True
    assert settled[0]["ai_brier"] == compute_brier_3class(0.7, 0.2, 0.1, "up")

    traj = compute_open_trajectory_status(records, taiex_ser)
    assert any(t["horizon_key"] == "1_3y" for t in traj)

    port = compute_shadow_portfolio(records, etf_ser, initial_capital=1_000_000.0)
    assert port["available"] is True
    assert port["strategies"]["ai"]["total_cost"] > 0
    assert port["strategies"]["buy_hold_0050"]["total_return_pct"] > 0
