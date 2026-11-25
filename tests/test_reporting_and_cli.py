"""週報、月報、儀表板產出與無簡體／中國用語規範測試。"""

from __future__ import annotations

from pathlib import Path

from twse_forecast.cli import main
from twse_forecast.config import HORIZONS, ProjectPaths
from twse_forecast.evaluation.audit import summarize_monthly_audit
from twse_forecast.models import MarketSeries, OHLCVBar
from twse_forecast.reporting.dashboard import render_dashboard_html
from twse_forecast.reporting.monthly_report import render_monthly_audit_report
from twse_forecast.reporting.weekly_report import render_weekly_report


def test_reports_and_dashboard_rendering(tmp_path: Path) -> None:
    q_h = {
        spec.key: {
            "horizon_key": spec.key,
            "label": spec.label,
            "anchor_trading_days": spec.anchor_trading_days,
            "quant_score": 15.0,
            "expected_return": 0.02,
            "std_dev": 0.04,
            "prob_up": 0.5,
            "prob_flat": 0.3,
            "prob_down": 0.2,
            "ci80_low_level": 21000.0,
            "ci80_high_level": 23000.0,
        }
        for spec in HORIZONS
    }
    ai_h = {
        spec.key: {
            "horizon_key": spec.key,
            "ai_score": 25.0,
            "quant_score": 15.0,
            "delta_score": 10.0,
            "expected_return": 0.03,
            "prob_up": 0.55,
            "prob_flat": 0.25,
            "prob_down": 0.20,
            "ci80_low_level": 21200.0,
            "ci80_high_level": 23400.0,
            "invalidation_level": 21000.0,
            "thesis": f"{spec.label} 維持溫和偏多",
        }
        for spec in HORIZONS
    }
    record = {
        "record_id": "2026-W48",
        "scheduled_slot": "2026-11-23T08:00:00+08:00",
        "executed_at_iso": "2026-11-24T21:30:00+08:00",
        "is_late_catchup": True,
        "reference_trade_date": "2026-11-24",
        "reference_taiex_close": 22500.0,
        "reference_0050_adj_close": 195.0,
        "prev_hash": "GENESIS",
        "record_hash": "a" * 64,
        "quant_baseline": {"target_allocation": 0.65, "horizons": q_h},
        "ai_forecast": {
            "model_name": "Gemini 4 Carbon",
            "target_allocation": 0.70,
            "bull_case": "半導體營收維持高檔",
            "bear_case": "美債殖利率高檔震盪",
            "cio_synthesis": "溫和偏多但控管部位",
            "override_rationale": "考量美股科技財報動能上調短線分數",
            "horizons": ai_h,
            "external_news_notes": ["聯準會紀要偏中性"],
        },
    }
    features = {
        "cross_market": {"tsmc_adr_premium_pct": 10.5},
        "fundamentals_valuation": {"market_median_pb": 1.8},
        "microstructure_flows": {"foreign_5d_net_b": 120.0},
        "technicals": {"ma60_deduction": {"regime": "助漲"}},
        "breadth": {"pct_above_ma60": 55.0},
        "sentiment": {"retail_euphoria_score": 5.0, "contrarian_signal": 0.0},
    }

    taiex_ser = MarketSeries("^TWII", [OHLCVBar("2026-11-24", 22400, 22600, 22300, 22500, 22500, 1000)])
    etf_ser = MarketSeries("0050.TW", [OHLCVBar("2026-11-24", 194, 196, 193, 195, 195, 1000)])
    audit_sum = summarize_monthly_audit([record], [], taiex_ser, etf_ser)

    weekly_md = render_weekly_report(record=record, features=features, audit_summary=audit_sum)
    assert "五週期預測承諾表" in weekly_md
    assert "台股大盤多週期預測週報" in weekly_md

    monthly_md = render_monthly_audit_report(
        audit_month="2026-11",
        audit_summary=audit_sum,
        ledger_valid=True,
        ledger_issues=[],
        recent_settlements=[],
    )
    assert "月度實證績效評估報告" in monthly_md

    dash_html = render_dashboard_html(
        records=[record],
        latest_features=features,
        audit_summary=audit_sum,
        ledger_valid=True,
        ledger_issues=[],
    )
    assert "SHA-256 雜湊鏈驗證通過" in dash_html
    assert "<!DOCTYPE html>" in dash_html

    # 測試空白帳本 verify-ledger CLI 通過
    paths = ProjectPaths(root=tmp_path)
    paths.ensure_dirs()
    try:
        main(["verify-ledger", "--root", str(tmp_path)])
    except SystemExit as exc:
        assert exc.code == 0
