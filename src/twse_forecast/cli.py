"""twse-forecast 命令列進入點：prepare-weekly / commit-weekly / monthly-audit / verify-ledger。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from twse_forecast.collectors.macro_fred import collect_fred_macro
from twse_forecast.collectors.market_yahoo import (
    collect_yahoo_market_data,
    fetch_single_yahoo_symbol,
)
from twse_forecast.collectors.sentiment_news import collect_sentiment_and_news
from twse_forecast.collectors.taifex_official import collect_taifex_data
from twse_forecast.collectors.twse_official import collect_twse_official_data
from twse_forecast.config import DEFAULT_PROJECT_ROOT, TAIPEI_TZ, ProjectPaths
from twse_forecast.engine.briefing import (
    render_briefing_markdown,
    save_json_file,
    validate_and_finalize_ai_submission,
)
from twse_forecast.engine.gemini_agent import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_THINKING_LEVEL,
    generate_ai_forecast,
)
from twse_forecast.engine.quant_baseline import run_quant_baseline
from twse_forecast.evaluation.audit import settle_matured_predictions, summarize_monthly_audit
from twse_forecast.features.feature_builder import build_feature_snapshot
from twse_forecast.ledger.store import (
    append_prediction_record,
    compute_schedule_metadata,
    load_all_settlements,
    load_ledger_records,
    save_settlements,
    verify_ledger,
)
from twse_forecast.models import MarketSeries, OHLCVBar, to_plain_dict
from twse_forecast.reporting.charts import generate_all_charts
from twse_forecast.reporting.dashboard import render_dashboard_html, write_dashboard_html
from twse_forecast.reporting.monthly_report import render_monthly_audit_report, write_monthly_report
from twse_forecast.reporting.weekly_report import render_weekly_report, write_weekly_report
from twse_forecast.sync.gdrive_sync import (
    has_gdrive_credentials_configured,
    sync_project_outputs_to_gdrive,
)


def _serialize_core_series(series_map: dict[str, MarketSeries]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for sym in ("^TWII", "0050.TW"):
        if sym in series_map:
            out[sym] = to_plain_dict(series_map[sym])
    return out


def _load_core_series(path: Path) -> dict[str, MarketSeries]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, MarketSeries] = {}
    for sym, obj in raw.items():
        bars = [OHLCVBar(**b) for b in obj["bars"]]
        out[sym] = MarketSeries(symbol=obj["symbol"], bars=bars)
    return out


def cmd_prepare_weekly(args: argparse.Namespace) -> int:
    paths = ProjectPaths(root=Path(args.root))
    paths.ensure_dirs()

    print("[1/5] 正在擷取 Yahoo Finance 跨市場與台股 30 檔代表籃子...", flush=True)
    series_map, yahoo_summary, r_yahoo = collect_yahoo_market_data()
    if r_yahoo.status == "failed":
        print(f"錯誤：無法取得核心指數 ^TWII 或 0050.TW：{r_yahoo.message}", file=sys.stderr, flush=True)
        return 1

    print("[2/5] 正在擷取證交所、期交所、FRED 與 PTT 股板／財經新聞...", flush=True)
    twse_payload, r_twse = collect_twse_official_data()
    taifex_payload, r_taifex = collect_taifex_data()
    fred_payload, r_fred = collect_fred_macro()
    sentiment_payload, r_sent = collect_sentiment_and_news()

    collector_reports = [to_plain_dict(r) for r in (r_yahoo, r_twse, r_taifex, r_fred, r_sent)]

    print("[3/5] 正在建置多週期特徵快照與計算第一層純量化基準...", flush=True)
    features = build_feature_snapshot(
        series_map=series_map,
        yahoo_summary=yahoo_summary,
        twse_payload=twse_payload,
        taifex_payload=taifex_payload,
        fred_payload=fred_payload,
        sentiment_payload=sentiment_payload,
    )
    quant_result_obj = run_quant_baseline(features)
    quant_result = to_plain_dict(quant_result_obj)

    print("[4/5] 正在檢查並結算已到期之歷史預測...")
    records = load_ledger_records(paths)
    existing_settlements = load_all_settlements(paths)
    existing_ids = {s["settlement_id"] for s in existing_settlements}
    new_settled = settle_matured_predictions(
        records, series_map["^TWII"], existing_settlement_ids=existing_ids
    )
    inserted = save_settlements(paths, new_settled)
    all_settlements = existing_settlements + new_settled

    schedule_meta = compute_schedule_metadata()
    record_id = args.record_id or schedule_meta["record_id"]

    # 儲存中繼快照
    save_json_file(paths.snapshots_dir / "latest_core_series.json", _serialize_core_series(series_map))
    save_json_file(paths.snapshots_dir / "latest_features.json", features)
    save_json_file(paths.snapshots_dir / "latest_quant_baseline.json", quant_result)
    save_json_file(paths.snapshots_dir / "latest_collector_reports.json", collector_reports)

    print("[5/5] 正在產出 AI 辯證閱讀簡報包...")
    briefing_md = render_briefing_markdown(
        record_id=record_id,
        features=features,
        quant_result=quant_result,
        recent_settled_reviews=all_settlements[-4:],
        collector_reports=collector_reports,
    )
    briefing_path = paths.snapshots_dir / "latest_briefing.md"
    briefing_path.write_text(briefing_md, encoding="utf-8")

    now_dt = datetime.now(TAIPEI_TZ)
    month_tag = now_dt.strftime("%Y-%m")
    monthly_report_path = paths.monthly_reports_dir / f"{month_tag}-audit.md"
    monthly_due = len(records) > 0 and now_dt.day <= 7 and not monthly_report_path.exists()

    print(f"\n完成！基準交易日：{features['reference_trade_date']}，TAIEX：{features['reference_taiex_close']:,.2f}")
    print(f"本次新結算預測筆數：{inserted}（累計已結算：{len(all_settlements)} 筆）")
    print(f"簡報檔案位置：{briefing_path}")
    print(f"MONTHLY_AUDIT_DUE={'true' if monthly_due else 'false'}")
    return 0


def cmd_commit_weekly(args: argparse.Namespace) -> int:
    paths = ProjectPaths(root=Path(args.root))
    paths.ensure_dirs()

    features_path = paths.snapshots_dir / "latest_features.json"
    quant_path = paths.snapshots_dir / "latest_quant_baseline.json"
    core_series_path = paths.snapshots_dir / "latest_core_series.json"
    raw_forecast_arg = Path(args.forecast_json)
    ai_json_path = raw_forecast_arg if raw_forecast_arg.is_absolute() else (paths.root / raw_forecast_arg)

    for p in (features_path, quant_path, core_series_path, ai_json_path):
        if not p.exists():
            print(f"錯誤：找不到必要輸入檔 {p}", file=sys.stderr)
            return 1

    features = json.loads(features_path.read_text(encoding="utf-8"))
    quant_result = json.loads(quant_path.read_text(encoding="utf-8"))
    core_series = _load_core_series(core_series_path)
    raw_ai = json.loads(ai_json_path.read_text(encoding="utf-8"))

    ai_submission = validate_and_finalize_ai_submission(raw_ai, quant_result, features)
    schedule_meta = compute_schedule_metadata()
    if args.record_id:
        schedule_meta["record_id"] = args.record_id

    record_id = schedule_meta["record_id"]
    snapshot_archive_path = paths.snapshots_dir / f"{record_id}_snapshot.json"
    save_json_file(snapshot_archive_path, features)

    record = append_prediction_record(
        paths,
        schedule_meta=schedule_meta,
        features=features,
        quant_baseline=quant_result,
        ai_forecast=to_plain_dict(ai_submission),
        snapshot_path=str(snapshot_archive_path.relative_to(paths.root)),
    )

    is_valid, issues, total = verify_ledger(paths)
    if not is_valid:
        print(f"嚴重錯誤：帳本寫入後驗證失敗：{issues}", file=sys.stderr)
        return 1

    records = load_ledger_records(paths)
    settlements = load_all_settlements(paths)
    audit_summary = summarize_monthly_audit(
        records, settlements, core_series["^TWII"], core_series["0050.TW"]
    )

    weekly_md = render_weekly_report(record=record, features=features, audit_summary=audit_summary)
    weekly_path = paths.weekly_reports_dir / f"{record_id}.md"
    write_weekly_report(weekly_path, weekly_md)

    dash_html = render_dashboard_html(
        records=records,
        latest_features=features,
        audit_summary=audit_summary,
        ledger_valid=is_valid,
        ledger_issues=issues,
    )
    write_dashboard_html(paths.dashboard_html, dash_html)
    generate_all_charts(
        charts_dir=paths.dashboard_html.parent / "charts",
        records=records,
        audit_summary=audit_summary,
    )

    print(f"已成功寫入不可竄改帳本：{record_id}（累計 {total} 筆）")
    print(f"SHA-256 雜湊碼：{record['record_hash']}")
    print(f"週報位置：{weekly_path}")
    print(f"儀表板位置：{paths.dashboard_html}")
    return 0


def cmd_monthly_audit(args: argparse.Namespace) -> int:
    paths = ProjectPaths(root=Path(args.root))
    paths.ensure_dirs()

    records = load_ledger_records(paths)
    core_series_path = paths.snapshots_dir / "latest_core_series.json"
    features_path = paths.snapshots_dir / "latest_features.json"

    core_series: dict[str, MarketSeries] = {}
    if not args.no_refresh:
        try:
            with httpx.Client(timeout=15.0, follow_redirects=True) as client:
                core_series["^TWII"] = fetch_single_yahoo_symbol(client, "^TWII", range_str="5y")
                core_series["0050.TW"] = fetch_single_yahoo_symbol(client, "0050.TW", range_str="5y")
            save_json_file(core_series_path, _serialize_core_series(core_series))
        except Exception as exc:  # noqa: BLE001
            print(f"提示：更新最新大盤價格失敗（{exc}），改用本地快取序列。")

    if not core_series:
        if not core_series_path.exists():
            print("錯誤：無可用核心行情序列，請先執行 prepare-weekly。", file=sys.stderr)
            return 1
        core_series = _load_core_series(core_series_path)

    existing_settlements = load_all_settlements(paths)
    existing_ids = {s["settlement_id"] for s in existing_settlements}
    new_settled = settle_matured_predictions(
        records, core_series["^TWII"], existing_settlement_ids=existing_ids
    )
    save_settlements(paths, new_settled)
    all_settlements = existing_settlements + new_settled

    is_valid, issues, _ = verify_ledger(paths)
    audit_summary = summarize_monthly_audit(
        records, all_settlements, core_series["^TWII"], core_series["0050.TW"]
    )
    audit_month = args.month or datetime.now(TAIPEI_TZ).strftime("%Y-%m")
    monthly_md = render_monthly_audit_report(
        audit_month=audit_month,
        audit_summary=audit_summary,
        ledger_valid=is_valid,
        ledger_issues=issues,
        recent_settlements=all_settlements,
    )
    out_path = paths.monthly_reports_dir / f"{audit_month}-audit.md"
    write_monthly_report(out_path, monthly_md)

    latest_features = (
        json.loads(features_path.read_text(encoding="utf-8"))
        if features_path.exists()
        else None
    )
    dash_html = render_dashboard_html(
        records=records,
        latest_features=latest_features,
        audit_summary=audit_summary,
        ledger_valid=is_valid,
        ledger_issues=issues,
    )
    write_dashboard_html(paths.dashboard_html, dash_html)
    generate_all_charts(
        charts_dir=paths.dashboard_html.parent / "charts",
        records=records,
        audit_summary=audit_summary,
    )

    print(f"月度實證稽核報告已產出：{out_path}")
    print(f"儀表板已同步更新：{paths.dashboard_html}")
    return 0


def cmd_verify_ledger(args: argparse.Namespace) -> int:
    paths = ProjectPaths(root=Path(args.root))
    is_valid, issues, total = verify_ledger(paths)
    if is_valid:
        print(f"PASS: 預測帳本 SHA-256 雜湊鏈與雙寫儲存驗證通過（共 {total} 筆紀錄）。")
        return 0
    print("FAIL: 預測帳本驗證失敗！偵測到以下異常：", file=sys.stderr)
    for issue in issues:
        print(f" - {issue}", file=sys.stderr)
    return 1


def cmd_generate_ai(args: argparse.Namespace) -> int:
    paths = ProjectPaths(root=Path(args.root))
    paths.ensure_dirs()

    briefing_path = paths.snapshots_dir / "latest_briefing.md"
    features_path = paths.snapshots_dir / "latest_features.json"
    quant_path = paths.snapshots_dir / "latest_quant_baseline.json"

    for p in (briefing_path, features_path, quant_path):
        if not p.exists():
            print(f"錯誤：找不到必要輸入檔 {p}，請先執行 prepare-weekly。", file=sys.stderr)
            return 1

    briefing_md = briefing_path.read_text(encoding="utf-8")
    features = json.loads(features_path.read_text(encoding="utf-8"))
    quant_result = json.loads(quant_path.read_text(encoding="utf-8"))

    model_name = (args.model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()
    thinking_level = (
        args.thinking_level or os.environ.get("GEMINI_THINKING_LEVEL") or DEFAULT_THINKING_LEVEL
    ).strip()

    print(f"正在呼叫 Gemini API（模型：{model_name}，思考強度：{thinking_level}）進行多空辯證...", flush=True)
    ai_dict = generate_ai_forecast(
        briefing_md=briefing_md,
        quant_result=quant_result,
        features=features,
        model_name=model_name,
        thinking_level=thinking_level,
    )

    raw_out = Path(args.output)
    out_path = raw_out if raw_out.is_absolute() else (paths.root / raw_out)
    save_json_file(out_path, ai_dict)
    print(f"AI 辯證預測 JSON 已儲存並通過邊界驗證：{out_path}")
    return 0


def cmd_sync_gdrive(args: argparse.Namespace) -> int:
    paths = ProjectPaths(root=Path(args.root))
    results = sync_project_outputs_to_gdrive(paths, folder_id=args.folder_id)
    if not results:
        print("無可同步之報表或圖表檔案。")
        return 0
    for item in results:
        print(f"[{item.action.upper()}] {item.drive_name} (id={item.file_id})")
    return 0


def cmd_run_weekly_cloud(args: argparse.Namespace) -> int:
    paths = ProjectPaths(root=Path(args.root))
    paths.ensure_dirs()

    schedule_meta = compute_schedule_metadata()
    target_record_id = args.record_id or schedule_meta["record_id"]

    if args.skip_if_exists:
        existing = load_ledger_records(paths)
        if any(r["record_id"] == target_record_id for r in existing):
            print(f"提示：帳本中已存在週次 {target_record_id}，依 --skip-if-exists 略過重複預測。")
            return cmd_verify_ledger(argparse.Namespace(root=args.root))

    rc = cmd_prepare_weekly(argparse.Namespace(root=args.root, record_id=target_record_id))
    if rc != 0:
        return rc

    default_rel_json = "data/snapshots/latest_ai_forecast.json"
    rc = cmd_generate_ai(
        argparse.Namespace(
            root=args.root,
            model=args.model,
            thinking_level=args.thinking_level,
            output=default_rel_json,
        )
    )
    if rc != 0:
        return rc

    rc = cmd_commit_weekly(
        argparse.Namespace(
            root=args.root,
            forecast_json=default_rel_json,
            record_id=target_record_id,
        )
    )
    if rc != 0:
        return rc

    now_dt = datetime.now(TAIPEI_TZ)
    month_tag = now_dt.strftime("%Y-%m")
    monthly_report_path = paths.monthly_reports_dir / f"{month_tag}-audit.md"
    should_audit = bool(args.force_monthly_audit) or (
        now_dt.day <= 7 and not monthly_report_path.exists()
    )
    if should_audit:
        rc = cmd_monthly_audit(
            argparse.Namespace(
                root=args.root,
                month=month_tag,
                no_refresh=True,
            )
        )
        if rc != 0:
            return rc

    if has_gdrive_credentials_configured():
        print("偵測到 Google Drive 同步設定，正在上傳最新週報、月報與視覺化圖表...", flush=True)
        rc_sync = cmd_sync_gdrive(argparse.Namespace(root=args.root, folder_id=None))
        if rc_sync != 0:
            return rc_sync
    else:
        print("提示：未設定 GDRIVE_FOLDER_ID 或憑證，略過 Google Drive 自動同步。")

    return cmd_verify_ledger(argparse.Namespace(root=args.root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="twse-forecast",
        description="台灣股市多週期 AI 與量化大盤預測及月度實證稽核系統",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare-weekly", help="擷取公開市場資料、結算到期預測並產出簡報包")
    p_prep.add_argument("--root", default=str(DEFAULT_PROJECT_ROOT), help="工作區根目錄")
    p_prep.add_argument("--record-id", default=None, help="指定週次代碼（預設自動依台灣時間計算）")
    p_prep.set_defaults(func=cmd_prepare_weekly)

    p_gen = sub.add_parser("generate-ai", help="呼叫 Gemini API（預設 gemini-4-neon）產出五週期辯證預測")
    p_gen.add_argument("--model", default=None, help="Gemini 模型代碼（預設 gemini-4-neon）")
    p_gen.add_argument("--thinking-level", default=None, help="思考強度 low / medium / high / xhigh")
    p_gen.add_argument("--output", default="data/snapshots/latest_ai_forecast.json", help="輸出 JSON 路徑")
    p_gen.add_argument("--root", default=str(DEFAULT_PROJECT_ROOT), help="工作區根目錄")
    p_gen.set_defaults(func=cmd_generate_ai)

    p_commit = sub.add_parser("commit-weekly", help="驗證 AI 預測並鎖定寫入 SHA-256 雜湊鏈帳本")
    p_commit.add_argument(
        "--forecast-json",
        default="data/snapshots/latest_ai_forecast.json",
        help="AI 辯證輸出 JSON 路徑",
    )
    p_commit.add_argument("--root", default=str(DEFAULT_PROJECT_ROOT), help="工作區根目錄")
    p_commit.add_argument("--record-id", default=None, help="指定週次代碼")
    p_commit.set_defaults(func=cmd_commit_weekly)

    p_audit = sub.add_parser("monthly-audit", help="執行月度實證績效評鑑與更新儀表板")
    p_audit.add_argument("--month", default=None, help="稽核月份（格式 YYYY-MM，預設當月）")
    p_audit.add_argument("--no-refresh", action="store_true", help="不連網更新大盤最新收盤，僅用本地快取")
    p_audit.add_argument("--root", default=str(DEFAULT_PROJECT_ROOT), help="工作區根目錄")
    p_audit.set_defaults(func=cmd_monthly_audit)

    p_ver = sub.add_parser("verify-ledger", help="驗證預測帳本 SHA-256 雜湊鏈完整性")
    p_ver.add_argument("--root", default=str(DEFAULT_PROJECT_ROOT), help="工作區根目錄")
    p_ver.set_defaults(func=cmd_verify_ledger)

    p_sync = sub.add_parser("sync-gdrive", help="將最新報表、視覺化圖表與帳本同步至 Google Drive")
    p_sync.add_argument("--folder-id", default=None, help="目標 Google Drive 資料夾 ID（預設讀 GDRIVE_FOLDER_ID）")
    p_sync.add_argument("--root", default=str(DEFAULT_PROJECT_ROOT), help="工作區根目錄")
    p_sync.set_defaults(func=cmd_sync_gdrive)

    p_cloud = sub.add_parser("run-weekly-cloud", help="雲端排程專用：一鍵執行資料擷取、Gemini 辯證、鎖定帳本與月度稽核")
    p_cloud.add_argument("--model", default=None, help="Gemini 模型代碼（預設 gemini-4-neon）")
    p_cloud.add_argument("--thinking-level", default=None, help="思考強度 low / medium / high / xhigh")
    p_cloud.add_argument("--record-id", default=None, help="指定週次代碼（預設自動計算）")
    p_cloud.add_argument("--force-monthly-audit", action="store_true", help="強制產出當月績效稽核報告")
    p_cloud.add_argument("--skip-if-exists", action="store_true", help="若當週已存在於帳本則安全跳過")
    p_cloud.add_argument("--root", default=str(DEFAULT_PROJECT_ROOT), help="工作區根目錄")
    p_cloud.set_defaults(func=cmd_run_weekly_cloud)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    rc = args.func(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
