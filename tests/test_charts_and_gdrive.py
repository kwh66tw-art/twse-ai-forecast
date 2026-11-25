"""高解析視覺化圖表與 Google Drive 同步單元測試。"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from twse_forecast.config import HORIZONS, ProjectPaths
from twse_forecast.evaluation.audit import summarize_monthly_audit
from twse_forecast.models import MarketSeries, OHLCVBar
from twse_forecast.reporting.charts import generate_all_charts
from twse_forecast.sync.gdrive_sync import (
    collect_sync_targets,
    has_gdrive_credentials_configured,
    sync_project_outputs_to_gdrive,
)


def _mock_record(rec_id: str, ref_date: str, ref_close: float, alloc: float) -> dict:
    return {
        "record_id": rec_id,
        "scheduled_slot": f"{ref_date}T08:00:00+08:00",
        "executed_at_iso": f"{ref_date}T08:01:00+08:00",
        "is_late_catchup": False,
        "reference_trade_date": ref_date,
        "reference_taiex_close": ref_close,
        "reference_0050_adj_close": ref_close / 400.0,
        "prev_hash": "GENESIS",
        "record_hash": "a" * 64,
        "quant_baseline": {
            "target_allocation": alloc,
            "horizons": {
                s.key: {
                    "quant_score": 15.0,
                    "expected_return": 0.03,
                    "std_dev": 0.05,
                    "prob_up": 0.5,
                    "prob_flat": 0.3,
                    "prob_down": 0.2,
                }
                for s in HORIZONS
            },
        },
        "ai_forecast": {
            "model_name": "gemini-4-neon",
            "target_allocation": alloc,
            "bull_case": "B",
            "bear_case": "R",
            "cio_synthesis": "S",
            "override_rationale": "O",
            "horizons": {
                s.key: {
                    "ai_score": 25.0,
                    "quant_score": 15.0,
                    "delta_score": 10.0,
                    "expected_return": 0.04,
                    "prob_up": 0.55,
                    "prob_flat": 0.25,
                    "prob_down": 0.20,
                    "ci80_low_level": ref_close * 0.92,
                    "ci80_high_level": ref_close * 1.12,
                    "invalidation_level": ref_close * 0.93,
                    "thesis": "測試",
                }
                for s in HORIZONS
            },
        },
    }


def test_generate_all_charts_single_and_multi_week(tmp_path: Path) -> None:
    r1 = _mock_record("2026-W48", "2026-11-24", 40000.0, 0.28)
    taiex = MarketSeries("^TWII", [OHLCVBar("2026-11-24", 40000, 40100, 39900, 40000, 40000, 100)])
    etf = MarketSeries("0050.TW", [OHLCVBar("2026-11-24", 100, 101, 99, 100, 100, 100)])

    # 情境一：單週基期、尚無到期結算
    summary_w1 = summarize_monthly_audit([r1], [], taiex, etf)
    out_w1 = generate_all_charts(charts_dir=tmp_path / "charts_w1", records=[r1], audit_summary=summary_w1)
    assert set(out_w1.keys()) == {"horizons", "performance", "scorecard"}
    for p in out_w1.values():
        assert p.exists()
        assert p.read_bytes()[:4] == b"\x89PNG"

    # 情境二：多週淨值變化 + 已到期結算計分卡
    r2 = _mock_record("2026-W49", "2026-12-01", 40800.0, 0.55)
    etf_multi = MarketSeries("0050.TW", [
        OHLCVBar("2026-11-24", 100, 101, 99, 100, 100, 100),
        OHLCVBar("2026-12-01", 102, 103, 101, 102, 102, 100),
    ])
    settled = [{
        "settlement_id": "2026-W48:1_4w",
        "record_id": "2026-W48",
        "horizon_key": "1_4w",
        "horizon_label": "1–4 週",
        "actual_return": 0.03,
        "actual_outcome": "up",
        "ai_expected_return": 0.04,
        "ai_hit": True,
        "quant_hit": True,
        "ai_brier": 0.18,
        "quant_brier": 0.24,
        "naive_bull_brier": 0.22,
        "in_ai_ci80": True,
        "in_quant_ci80": True,
    }]
    summary_multi = summarize_monthly_audit([r1, r2], settled, taiex, etf_multi)
    out_m = generate_all_charts(charts_dir=tmp_path / "charts_m", records=[r1, r2], audit_summary=summary_multi)
    for p in out_m.values():
        assert p.exists()
        assert p.read_bytes()[:4] == b"\x89PNG"


def test_gdrive_sync_upsert_and_missing_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GDRIVE_FOLDER_ID", raising=False)
    monkeypatch.delenv("GDRIVE_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("GDRIVE_SERVICE_ACCOUNT_JSON", raising=False)
    assert has_gdrive_credentials_configured() is False

    paths = ProjectPaths(root=tmp_path)
    paths.ensure_dirs()
    (paths.weekly_reports_dir / "2026-W48.md").write_text("# 週報", encoding="utf-8")
    (paths.monthly_reports_dir / "2026-11-audit.md").write_text("# 月報", encoding="utf-8")
    charts_dir = paths.dashboard_html.parent / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    (charts_dir / "horizons_forecast.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    paths.dashboard_html.write_text("<html></html>", encoding="utf-8")
    paths.ledger_jsonl.write_text("{}\n", encoding="utf-8")

    targets = collect_sync_targets(paths)
    assert len(targets) == 5

    # Mock Transport：模擬第一個檔案已存在（觸發 PATCH 更新），其餘不存在（觸發 POST 建立）
    requests_seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append((request.method, str(request.url)))
        if request.method == "GET" and "/drive/v3/files" in str(request.url):
            q = request.url.params.get("q", "")
            if "台股預測週報_2026-W48" in q:
                return httpx.Response(200, json={"files": [{"id": "existing_doc_1", "name": "台股預測週報_2026-W48"}]})
            return httpx.Response(200, json={"files": []})
        if request.method == "PATCH":
            return httpx.Response(200, json={"id": "existing_doc_1"})
        if request.method == "POST":
            body_text = request.content.decode("utf-8", errors="ignore")
            if "台股月度績效稽核" in body_text:
                assert "application/vnd.google-apps.document" in body_text
            return httpx.Response(200, json={"id": f"new_id_{len(requests_seen)}"})
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as mock_client:
        results = sync_project_outputs_to_gdrive(
            paths,
            folder_id="folder_123",
            access_token="fake_token",
            http_client=mock_client,
        )

    assert len(results) == 5
    actions = {r.drive_name: r.action for r in results}
    assert actions["台股預測週報_2026-W48"] == "updated"
    assert actions["01_五週期預測機率與點位通道.png"] == "created"
    assert actions["台股預測互動儀表板.html"] == "created"
