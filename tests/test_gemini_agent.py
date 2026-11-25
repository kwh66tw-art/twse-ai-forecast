"""Gemini 4 Neon 雲端辯證引擎與 run-weekly-cloud CLI 整合測試。"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from twse_forecast.cli import cmd_run_weekly_cloud
from twse_forecast.config import HORIZONS, ProjectPaths
from twse_forecast.engine.gemini_agent import (
    GeminiForecastSchema,
    HorizonPredictionSchema,
    generate_ai_forecast,
)
from twse_forecast.ledger.store import append_prediction_record


def _sample_quant_and_features() -> tuple[dict[str, Any], dict[str, Any]]:
    q_horizons = {
        spec.key: {
            "horizon_key": spec.key,
            "label": spec.label,
            "anchor_trading_days": spec.anchor_trading_days,
            "quant_score": 10.0,
            "expected_return": 0.02,
            "std_dev": 0.05,
            "prob_up": 0.5,
            "prob_flat": 0.3,
            "prob_down": 0.2,
            "ci80_low_return": -0.04,
            "ci80_high_return": 0.08,
            "ci80_low_level": 38500.0,
            "ci80_high_level": 43300.0,
            "key_drivers": {},
        }
        for spec in HORIZONS
    }
    quant = {
        "reference_trade_date": "2026-11-24",
        "reference_taiex_close": 40000.0,
        "target_allocation": 0.50,
        "horizons": q_horizons,
    }
    features = {
        "reference_trade_date": "2026-11-24",
        "reference_taiex_close": 40000.0,
        "reference_0050_adj_close": 95.0,
        "technicals": {"realized_vol_20d": 0.20},
        "historical_horizon_stats": {
            spec.key: {"mean_return": 0.02, "std_return": 0.05}
            for spec in HORIZONS
        },
    }
    return quant, features


def _sample_schema(delta_1_4w: float = 15.0) -> GeminiForecastSchema:
    return GeminiForecastSchema(
        bull_case="營收正成長廣度維持高檔。",
        bear_case="美債殖利率偏高壓抑科技股估值。",
        cio_synthesis="採分批布局與波動率控管。",
        override_rationale="考量美股最新動能適度微調短線分數。",
        target_allocation=0.55,
        external_news_notes=["聯準會官員談話維持偏鷹基調"],
        horizon_1_4w=HorizonPredictionSchema(
            ai_score=10.0 + delta_1_4w,
            invalidation_level=38500.0,
            thesis="短線消化上檔賣壓。",
        ),
        horizon_1_3m=HorizonPredictionSchema(
            ai_score=20.0,
            invalidation_level=38000.0,
            thesis="月營收動能支撐季線修復。",
        ),
        horizon_3_6m=HorizonPredictionSchema(
            ai_score=25.0,
            invalidation_level=37500.0,
            thesis="中期 AI 訂單能見度高。",
        ),
        horizon_6_12m=HorizonPredictionSchema(
            ai_score=30.0,
            invalidation_level=37000.0,
            thesis="全年獲利仍具成長性。",
        ),
        horizon_1_3y=HorizonPredictionSchema(
            ai_score=28.0,
            invalidation_level=35000.0,
            thesis="長期估值與結構性護城河穩健。",
        ),
    )


class _FakeModelsPrimary:
    def __init__(self, parsed_obj: GeminiForecastSchema) -> None:
        self.calls: list[dict[str, Any]] = []
        self._parsed = parsed_obj

    def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        chunk = SimpleNamespace(web=SimpleNamespace(title="工商時報", uri="https://ctee.com.tw/news"))
        cand = SimpleNamespace(grounding_metadata=SimpleNamespace(grounding_chunks=[chunk]))
        return SimpleNamespace(
            parsed=self._parsed,
            text=self._parsed.model_dump_json(),
            candidates=[cand],
        )


class _FakeModelsTwoStageFallback:
    def __init__(self, parsed_obj: GeminiForecastSchema) -> None:
        self.calls: list[dict[str, Any]] = []
        self._parsed = parsed_obj

    def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        has_tools = bool(getattr(config, "tools", None))
        has_schema = getattr(config, "response_schema", None) is not None
        if has_tools and has_schema:
            raise ValueError("400 INVALID_ARGUMENT: response_schema cannot be used with google_search tool")
        if has_tools:
            chunk = SimpleNamespace(web=SimpleNamespace(title="自由財經", uri="https://ec.ltn.com.tw"))
            cand = SimpleNamespace(grounding_metadata=SimpleNamespace(grounding_chunks=[chunk]))
            return SimpleNamespace(
                parsed=None,
                text="研究備忘錄：多頭基本面佳，空頭留意美債利率。",
                candidates=[cand],
            )
        return SimpleNamespace(
            parsed=self._parsed,
            text=self._parsed.model_dump_json(),
            candidates=[],
        )


def test_gemini_primary_path_and_bounded_override() -> None:
    quant, features = _sample_quant_and_features()
    # 刻意給 1_4w 超出 +60 分，確認後端強制截斷在 +25 分內
    fake_models = _FakeModelsPrimary(_sample_schema(delta_1_4w=60.0))
    fake_client = SimpleNamespace(models=fake_models)

    res = generate_ai_forecast(
        briefing_md="# 測試簡報",
        quant_result=quant,
        features=features,
        client=fake_client,
    )

    assert len(fake_models.calls) == 1
    call = fake_models.calls[0]
    assert call["model"] == "gemini-4-neon"
    assert call["config"].thinking_config.thinking_level == "HIGH" or str(call["config"].thinking_config.thinking_level).lower().endswith("high")
    assert res["model_name"] == "gemini-4-neon"
    assert res["horizons"]["1_4w"]["delta_score"] == pytest.approx(25.0, abs=1e-4)
    assert any("工商時報" in n for n in res["external_news_notes"])


def test_gemini_two_stage_fallback_on_tool_schema_conflict() -> None:
    quant, features = _sample_quant_and_features()
    fake_models = _FakeModelsTwoStageFallback(_sample_schema(delta_1_4w=12.0))
    fake_client = SimpleNamespace(models=fake_models)

    res = generate_ai_forecast(
        briefing_md="# 測試簡報",
        quant_result=quant,
        features=features,
        client=fake_client,
    )

    assert len(fake_models.calls) == 3  # 1 次主路徑衝突 + 2 次降級
    assert res["horizons"]["1_4w"]["ai_score"] == 22.0
    assert any("自由財經" in n for n in res["external_news_notes"])


def test_gemini_missing_api_key_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    quant, features = _sample_quant_and_features()
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        generate_ai_forecast(
            briefing_md="# 測試簡報",
            quant_result=quant,
            features=features,
            client=None,
        )


def test_run_weekly_cloud_skip_if_exists(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    quant, features = _sample_quant_and_features()
    ai_dict = {
        "model_name": "gemini-4-neon",
        "bull_case": "B",
        "bear_case": "R",
        "cio_synthesis": "S",
        "override_rationale": "O",
        "target_allocation": 0.5,
        "horizons": {
            spec.key: {
                "horizon_key": spec.key,
                "ai_score": 15.0,
                "quant_score": 10.0,
                "delta_score": 5.0,
                "expected_return": 0.02,
                "prob_up": 0.5,
                "prob_flat": 0.3,
                "prob_down": 0.2,
                "ci80_low_level": 38500.0,
                "ci80_high_level": 43300.0,
                "invalidation_level": 38500.0,
                "thesis": "測試",
            }
            for spec in HORIZONS
        },
    }
    append_prediction_record(
        paths,
        schedule_meta={
            "record_id": "2026-W48",
            "scheduled_slot": "2026-11-23T08:00:00+08:00",
            "executed_at_iso": "2026-11-23T08:02:00+08:00",
            "is_late_catchup": False,
        },
        features=features,
        quant_baseline=quant,
        ai_forecast=ai_dict,
        snapshot_path="data/snapshots/2026-W48_snapshot.json",
    )

    rc = cmd_run_weekly_cloud(
        argparse.Namespace(
            root=str(tmp_path),
            model=None,
            thinking_level=None,
            record_id="2026-W48",
            force_monthly_audit=False,
            skip_if_exists=True,
        )
    )
    assert rc == 0
