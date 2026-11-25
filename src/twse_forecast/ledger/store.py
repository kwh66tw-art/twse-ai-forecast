"""防竄改預測帳本：雙寫 SQLite 與 Append-Only JSONL，並以 SHA-256 雜湊鏈驗證完整性。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from twse_forecast.config import TAIPEI_TZ, ProjectPaths


def canonical_json_dumps(obj: Any) -> str:
    """產生確定性、不可變鍵順序之 JSON 字串供雜湊計算。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_record_hash(prev_hash: str, core_payload: dict[str, Any]) -> str:
    """依 prev_hash 與核心預測內容計算 SHA-256 驗證碼。"""
    raw = f"{prev_hash}:{canonical_json_dumps(core_payload)}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def compute_schedule_metadata(now_taipei: datetime | None = None) -> dict[str, Any]:
    """計算台灣時間對應之週度排程槽位，並誠實判定是否屬於盤後／補跑紀錄。"""
    dt = now_taipei.astimezone(TAIPEI_TZ) if now_taipei else datetime.now(TAIPEI_TZ)
    # 找出本週一（weekday() == 0）08:00 槽位
    days_since_monday = dt.weekday()
    monday_date = (dt - timedelta(days=days_since_monday)).date()
    scheduled_dt = datetime(monday_date.year, monday_date.month, monday_date.day, 8, 0, 0, tzinfo=TAIPEI_TZ)
    iso_year, iso_week, _ = dt.isocalendar()
    record_id = f"{iso_year}-W{iso_week:02d}"

    # 週一 07:50 ~ 09:00（台股開盤前）視為標準盤前排程時段；其餘時段標記為非標準／補跑時段
    is_monday_premarket = (
        dt.weekday() == 0
        and (dt.hour == 7 and dt.minute >= 50 or dt.hour == 8)
    )
    return {
        "record_id": record_id,
        "scheduled_slot": scheduled_dt.isoformat(timespec="seconds"),
        "executed_at_iso": dt.isoformat(timespec="seconds"),
        "is_late_catchup": not is_monday_premarket,
    }


def init_sqlite(paths: ProjectPaths) -> sqlite3.Connection:
    """初始化帳本與結算 SQLite 資料表。"""
    paths.ensure_dirs()
    conn = sqlite3.connect(paths.ledger_sqlite)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS weekly_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id TEXT UNIQUE NOT NULL,
            scheduled_slot TEXT NOT NULL,
            executed_at_iso TEXT NOT NULL,
            is_late_catchup INTEGER NOT NULL,
            reference_trade_date TEXT NOT NULL,
            reference_taiex_close REAL NOT NULL,
            reference_0050_adj_close REAL NOT NULL,
            prev_hash TEXT NOT NULL,
            record_hash TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS horizon_settlements (
            settlement_id TEXT PRIMARY KEY,
            record_id TEXT NOT NULL,
            horizon_key TEXT NOT NULL,
            settled_at_iso TEXT NOT NULL,
            target_trade_date TEXT NOT NULL,
            target_taiex_close REAL NOT NULL,
            actual_return REAL NOT NULL,
            actual_outcome TEXT NOT NULL,
            ai_hit INTEGER NOT NULL,
            quant_hit INTEGER NOT NULL,
            ai_brier REAL NOT NULL,
            quant_brier REAL NOT NULL,
            in_ai_ci80 INTEGER NOT NULL,
            in_quant_ci80 INTEGER NOT NULL,
            settlement_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def load_ledger_records(paths: ProjectPaths) -> list[dict[str, Any]]:
    """從 JSONL 讀取所有已鎖定之歷史週度預測（若不存在回傳空清單）。"""
    if not paths.ledger_jsonl.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in paths.ledger_jsonl.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        records.append(json.loads(text))
    return records


def append_prediction_record(
    paths: ProjectPaths,
    *,
    schedule_meta: dict[str, Any],
    features: dict[str, Any],
    quant_baseline: dict[str, Any],
    ai_forecast: dict[str, Any],
    snapshot_path: str,
) -> dict[str, Any]:
    """將一筆完整週度預測附加寫入不可竄改 JSONL 與 SQLite 帳本。"""
    paths.ensure_dirs()
    existing = load_ledger_records(paths)
    record_id = schedule_meta["record_id"]
    if any(r["record_id"] == record_id for r in existing):
        raise ValueError(
            f"週次 {record_id} 已存在於不可竄改帳本中，嚴禁覆寫歷史預測！若為手動測試請使用獨立 record_id。"
        )

    prev_hash = existing[-1]["record_hash"] if existing else "GENESIS"
    core_payload: dict[str, Any] = {
        "record_id": record_id,
        "scheduled_slot": schedule_meta["scheduled_slot"],
        "executed_at_iso": schedule_meta["executed_at_iso"],
        "is_late_catchup": bool(schedule_meta["is_late_catchup"]),
        "reference_trade_date": features["reference_trade_date"],
        "reference_taiex_close": float(features["reference_taiex_close"]),
        "reference_0050_adj_close": float(features["reference_0050_adj_close"]),
        "snapshot_path": snapshot_path,
        "quant_baseline": quant_baseline,
        "ai_forecast": ai_forecast,
    }
    rec_hash = compute_record_hash(prev_hash, core_payload)
    full_record = {
        **core_payload,
        "prev_hash": prev_hash,
        "record_hash": rec_hash,
    }

    # 1. Append 至 JSONL
    with paths.ledger_jsonl.open("a", encoding="utf-8") as f:
        f.write(canonical_json_dumps(full_record) + "\n")

    # 2. 寫入 SQLite
    conn = init_sqlite(paths)
    try:
        conn.execute(
            """
            INSERT INTO weekly_predictions (
                record_id, scheduled_slot, executed_at_iso, is_late_catchup,
                reference_trade_date, reference_taiex_close, reference_0050_adj_close,
                prev_hash, record_hash, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                full_record["scheduled_slot"],
                full_record["executed_at_iso"],
                1 if full_record["is_late_catchup"] else 0,
                full_record["reference_trade_date"],
                full_record["reference_taiex_close"],
                full_record["reference_0050_adj_close"],
                prev_hash,
                rec_hash,
                canonical_json_dumps(full_record),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return full_record


def save_settlements(paths: ProjectPaths, settlements: list[dict[str, Any]]) -> int:
    """儲存新到期結算紀錄至 SQLite（已存在之 settlement_id 跳過以保持冪等）。"""
    if not settlements:
        return 0
    conn = init_sqlite(paths)
    inserted = 0
    try:
        for s in settlements:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO horizon_settlements (
                    settlement_id, record_id, horizon_key, settled_at_iso,
                    target_trade_date, target_taiex_close, actual_return,
                    actual_outcome, ai_hit, quant_hit, ai_brier, quant_brier,
                    in_ai_ci80, in_quant_ci80, settlement_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    s["settlement_id"],
                    s["record_id"],
                    s["horizon_key"],
                    s["settled_at_iso"],
                    s["target_trade_date"],
                    s["target_taiex_close"],
                    s["actual_return"],
                    s["actual_outcome"],
                    1 if s["ai_hit"] else 0,
                    1 if s["quant_hit"] else 0,
                    s["ai_brier"],
                    s["quant_brier"],
                    1 if s["in_ai_ci80"] else 0,
                    1 if s["in_quant_ci80"] else 0,
                    canonical_json_dumps(s),
                ),
            )
            inserted += cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return inserted


def load_all_settlements(paths: ProjectPaths) -> list[dict[str, Any]]:
    """讀取所有已到期結算紀錄。"""
    if not paths.ledger_sqlite.exists():
        return []
    conn = init_sqlite(paths)
    try:
        rows = conn.execute(
            "SELECT settlement_json FROM horizon_settlements ORDER BY target_trade_date ASC, record_id ASC"
        ).fetchall()
        return [json.loads(r["settlement_json"]) for r in rows]
    finally:
        conn.close()


def verify_ledger(paths: ProjectPaths) -> tuple[bool, list[str], int]:
    """嚴格驗證 JSONL 與 SQLite 雜湊鏈完整性，偵測任何事後竄改。"""
    issues: list[str] = []
    records = load_ledger_records(paths)
    if not records:
        if paths.ledger_sqlite.exists():
            conn = init_sqlite(paths)
            try:
                cnt = conn.execute("SELECT COUNT(*) FROM weekly_predictions").fetchone()[0]
                if cnt > 0:
                    issues.append(f"JSONL 為空，但 SQLite 卻有 {cnt} 筆紀錄，雙重儲存不一致。")
            finally:
                conn.close()
        return len(issues) == 0, issues, 0

    expected_prev = "GENESIS"
    for idx, rec in enumerate(records):
        rec_id = rec.get("record_id", f"index#{idx}")
        if rec.get("prev_hash") != expected_prev:
            issues.append(
                f"紀錄 {rec_id} 的 prev_hash 不連續（預期 {expected_prev}，實際 {rec.get('prev_hash')}）"
            )
        core = {k: v for k, v in rec.items() if k not in {"prev_hash", "record_hash"}}
        recomputed = compute_record_hash(rec.get("prev_hash", ""), core)
        if rec.get("record_hash") != recomputed:
            issues.append(
                f"紀錄 {rec_id} 的 SHA-256 雜湊不相符（疑似遭事後竄改！鎖定值 {rec.get('record_hash')} != 重算值 {recomputed}）"
            )
        expected_prev = rec.get("record_hash", "")

    # 交叉比對 SQLite 與 JSONL 是否一致
    if not paths.ledger_sqlite.exists():
        issues.append("遺失 SQLite 帳本檔案（ledger.sqlite）。")
    else:
        conn = init_sqlite(paths)
        try:
            db_rows = conn.execute(
                "SELECT record_id, record_hash FROM weekly_predictions ORDER BY id ASC"
            ).fetchall()
            if len(db_rows) != len(records):
                issues.append(f"SQLite 筆數（{len(db_rows)}）與 JSONL 筆數（{len(records)}）不一致。")
            else:
                for db_r, jl_r in zip(db_rows, records, strict=True):
                    if db_r["record_id"] != jl_r["record_id"] or db_r["record_hash"] != jl_r["record_hash"]:
                        issues.append(f"SQLite 與 JSONL 在 {jl_r['record_id']} 的雜湊不一致。")
        finally:
            conn.close()

    return len(issues) == 0, issues, len(records)
