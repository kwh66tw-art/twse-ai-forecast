"""第二層雲端辯證引擎：整合 Google GenAI SDK（預設 gemini-4-neon）與 Google Search Grounding。"""

from __future__ import annotations

import os
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from twse_forecast.config import HORIZONS, MAX_AI_SCORE_DELTA
from twse_forecast.engine.briefing import validate_and_finalize_ai_submission
from twse_forecast.models import to_plain_dict

DEFAULT_GEMINI_MODEL = "gemini-4-neon"
DEFAULT_THINKING_LEVEL = "high"
VALID_THINKING_LEVELS = {"low", "medium", "high", "xhigh"}


class HorizonPredictionSchema(BaseModel):
    """單一預測週期之結構化輸出模型。"""

    ai_score: float = Field(
        description="AI 調整後趨勢分數（-100 至 +100；與該週期純量化基準 quant_score 差距限制在 ±25 分內）"
    )
    invalidation_level: float = Field(
        description="該週期技術或結構失效關鍵點位（台灣加權指數點數）"
    )
    thesis: str = Field(
        description="該週期核心台灣中文論述，須具體引用簡報數據、新聞催化或籌碼／總經轉折依據"
    )


class GeminiForecastSchema(BaseModel):
    """Gemini API 多視角辯證與五週期預測承諾之 Pydantic 輸出結構。"""

    bull_case: str = Field(description="多頭代理人論點（綜合順風基本面、營收、資金動能與催化劑）")
    bear_case: str = Field(description="空頭稽核員論點（針對估值、利率、外資籌碼、關稅／地緣與技術反壓壓力測試）")
    cio_synthesis: str = Field(description="投資長綜合裁決（總結多空權衡與風險控管原則）")
    override_rationale: str = Field(
        description="說明各週期 AI 分數相較純量化基準調整的具體原因，若調降或調升皆須提出實證依據"
    )
    target_allocation: float = Field(
        ge=0.0,
        le=1.0,
        description="建議模擬持股水位（0.0 至 1.0；須參考波動率管理基準不宜過度偏離）",
    )
    external_news_notes: list[str] = Field(
        default_factory=list,
        description="透過即時搜尋查核到之重要週末／盤前突發消息或事件摘錄（純台灣中文）",
    )
    horizon_1_4w: HorizonPredictionSchema = Field(description="1–4 週（主驗算錨點 20 交易日）預測")
    horizon_1_3m: HorizonPredictionSchema = Field(description="1–3 個月（主驗算錨點 40 交易日）預測")
    horizon_3_6m: HorizonPredictionSchema = Field(description="3–6 個月（主驗算錨點 90 交易日）預測")
    horizon_6_12m: HorizonPredictionSchema = Field(description="6–12 個月（主驗算錨點 180 交易日）預測")
    horizon_1_3y: HorizonPredictionSchema = Field(description="1–3 年（主驗算錨點 375 交易日）預測")


def schema_to_submission_payload(
    parsed: GeminiForecastSchema,
    model_name: str,
    grounding_notes: list[str] | None = None,
) -> dict[str, Any]:
    """將 Gemini Pydantic 輸出轉換為系統標準提交格式。"""
    notes = [str(x).strip() for x in parsed.external_news_notes if str(x).strip()]
    for g in grounding_notes or []:
        if g not in notes and len(notes) < 10:
            notes.append(g)

    return {
        "model_name": model_name,
        "bull_case": parsed.bull_case,
        "bear_case": parsed.bear_case,
        "cio_synthesis": parsed.cio_synthesis,
        "override_rationale": parsed.override_rationale,
        "target_allocation": float(parsed.target_allocation),
        "external_news_notes": notes,
        "horizons": {
            "1_4w": parsed.horizon_1_4w.model_dump(),
            "1_3m": parsed.horizon_1_3m.model_dump(),
            "3_6m": parsed.horizon_3_6m.model_dump(),
            "6_12m": parsed.horizon_6_12m.model_dump(),
            "1_3y": parsed.horizon_1_3y.model_dump(),
        },
    }


def _extract_grounding_notes(response: Any, limit: int = 5) -> list[str]:
    """從 Gemini grounding_metadata 擷取實際檢索參考來源摘要。"""
    notes: list[str] = []
    seen: set[str] = set()
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return notes

    metadata = getattr(candidates[0], "grounding_metadata", None)
    if metadata is None:
        return notes

    chunks = (
        metadata.get("grounding_chunks")
        if isinstance(metadata, dict)
        else getattr(metadata, "grounding_chunks", None)
    ) or []

    for ch in chunks:
        web = ch.get("web") if isinstance(ch, dict) else getattr(ch, "web", None)
        if not web:
            continue
        title = (web.get("title") if isinstance(web, dict) else getattr(web, "title", "")) or ""
        uri = (web.get("uri") if isinstance(web, dict) else getattr(web, "uri", "")) or ""
        label = f"搜尋佐證：{title.strip()} ({uri.strip()})" if title and uri else (title or uri).strip()
        if label and label not in seen:
            seen.add(label)
            notes.append(label)
        if len(notes) >= limit:
            break
    return notes


def _build_prompts(
    briefing_md: str,
    quant_result: dict[str, Any],
    calibration_profile: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """組裝台灣中文系統提示詞與包含定量簡報之使用者提示詞。"""
    system_instruction = (
        "你是台灣股市大型法人研究團隊的投資長（CIO），必須同時綜合「多頭代理人」、"
        "「空頭稽核員」與「投資長裁決」三個嚴謹視角，產出台灣加權股價指數（TAIEX）"
        "五個時間週期的結構化趨勢預測。\n\n"
        "嚴格規範：\n"
        "1. 全程必須使用純正台灣中文與財經用語；嚴禁任何簡體字及中國詞彙"
        "（例如嚴禁：板塊、倉位、止損、止盈、市盈率、回撤、信息、質量、周期；"
        "應改用：族群／類股、部位／持股水位、停損、停利、本益比、最大回檔幅度、資訊、品質、週期）。\n"
        "2. 絕對不得編造或竄改第一層簡報中的官方指數收盤點位、估值、籌碼或歷史誤差數據。"
        "若即時搜尋結果中的新聞點位與簡報基準日官方收盤不同，以簡報基準官方收盤為基準。\n"
        f"3. 每個週期（1_4w、1_3m、3_6m、6_12m、1_3y）的 ai_score 與簡報中的 quant_score 差距"
        f"必須嚴格控制在下方指定之各週期合法調整上限內（預設 ±{MAX_AI_SCORE_DELTA:.0f} 分；"
        "若歷史主觀調整落後量化，系統會自動收緊上限），調整必須有清楚的因果證據"
        "（不可單純附和量化分數，亦不可過度自信）。\n"
        "4. 必須詳閱簡報第四節之「滾動偏差與自我校準診斷表」及「近期高信心誤判警示」："
        "若系統標註過往存在系統性偏多高估或偏空低估，本週裁決必須主動收斂偏差並檢討過往誤判。\n"
        "5. 必須明確給出每個週期的 invalidation_level（失效關鍵點位）與整體 target_allocation（0.0~1.0）。"
    )

    q_horizons = quant_result.get("horizons") or {}
    cal_by_h = (calibration_profile or {}).get("by_horizon") or {}
    bounds_lines = []
    for spec in HORIZONS:
        q_score = float((q_horizons.get(spec.key) or {}).get("quant_score", 0.0))
        h_max_delta = float((cal_by_h.get(spec.key) or {}).get("max_ai_delta", MAX_AI_SCORE_DELTA))
        low_b = max(-100.0, q_score - h_max_delta)
        high_b = min(100.0, q_score + h_max_delta)
        bounds_lines.append(
            f"- {spec.key}（{spec.label}）：quant_score = {q_score:+.2f}，"
            f"本週允許調整上限 ±{h_max_delta:.0f} 分，合法 ai_score 範圍 [{low_b:+.2f}, {high_b:+.2f}]"
        )

    user_prompt = (
        "請先使用 Google Search 查核過去 48 小時至今日影響台灣股市開盤與中長線展望的關鍵新聞"
        "（包含：美股與費半收盤／期貨、聯準會利率談話、美國關稅或地緣政策、台積電與 AI 供應鏈最新動態），"
        "再結合以下系統產出之完整量化與籌碼簡報包，完成多空辯證並輸出結構化預測：\n\n"
        "【各週期純量化基準與 AI 調整合法邊界】\n"
        + "\n".join(bounds_lines)
        + f"\n- 純量化建議持股水位（波動率管理後）：{float(quant_result.get('target_allocation', 0.5))*100:.1f}%\n\n"
        "【第一層系統簡報包全文】\n"
        + briefing_md
    )
    return system_instruction, user_prompt


def _parse_structured_response(response: Any) -> GeminiForecastSchema:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, GeminiForecastSchema):
        return parsed
    if isinstance(parsed, dict):
        return GeminiForecastSchema.model_validate(parsed)
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini API 回傳空白內容，無法解析為預測 JSON。")
    return GeminiForecastSchema.model_validate_json(text)


def _is_tool_schema_conflict(exc: Exception) -> bool:
    msg = str(exc).lower()
    keywords = ("invalid_argument", "400", "response_schema", "tool", "google_search", "json")
    return any(k in msg for k in keywords)


def generate_ai_forecast(
    *,
    briefing_md: str,
    quant_result: dict[str, Any],
    features: dict[str, Any],
    model_name: str = DEFAULT_GEMINI_MODEL,
    thinking_level: str = DEFAULT_THINKING_LEVEL,
    calibration_profile: dict[str, Any] | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """呼叫 Gemini API（預設 gemini-4-neon）產生經受限驗證之五週期預測 JSON。"""
    normalized_level = thinking_level.strip().lower()
    if normalized_level not in VALID_THINKING_LEVELS:
        raise ValueError(
            f"不支援的 thinking_level：{thinking_level}（僅支援：{', '.join(sorted(VALID_THINKING_LEVELS))}）"
        )

    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "找不到環境變數 GEMINI_API_KEY，無法呼叫 Gemini API。"
                "請於本機環境或 GitHub Repository Secrets 設定 GEMINI_API_KEY。"
            )
        client = genai.Client(api_key=api_key)

    system_instruction, user_prompt = _build_prompts(
        briefing_md, quant_result, calibration_profile=calibration_profile
    )
    grounding_notes: list[str] = []

    primary_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        thinking_config=types.ThinkingConfig(thinking_level=normalized_level),
        response_mime_type="application/json",
        response_schema=GeminiForecastSchema,
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=primary_config,
        )
        grounding_notes = _extract_grounding_notes(response)
        parsed = _parse_structured_response(response)
    except Exception as primary_exc:
        if not _is_tool_schema_conflict(primary_exc):
            raise
        # 兩階段降級備援：先開 Google Search + Thinking 產出研究備忘錄，再轉為嚴格 Schema JSON
        research_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            thinking_config=types.ThinkingConfig(thinking_level=normalized_level),
        )
        research_resp = client.models.generate_content(
            model=model_name,
            contents=user_prompt + "\n\n請先產出完整的多頭、空頭、投資長辯證與五週期具體點位／分數研究備忘錄。",
            config=research_config,
        )
        grounding_notes = _extract_grounding_notes(research_resp)
        research_text = getattr(research_resp, "text", None) or ""
        if not research_text.strip():
            raise RuntimeError(f"兩階段降級第一階段未取得有效文字回應；原始錯誤：{primary_exc}") from primary_exc

        json_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=GeminiForecastSchema,
        )
        json_resp = client.models.generate_content(
            model=model_name,
            contents=(
                "請將以下台灣股市多空辯證研究備忘錄嚴格轉換為指定 JSON 結構，"
                "務必保留所有具體點位與各週期合法調整上限：\n\n" + research_text
            ),
            config=json_config,
        )
        parsed = _parse_structured_response(json_resp)

    raw_payload = schema_to_submission_payload(
        parsed=parsed,
        model_name=model_name,
        grounding_notes=grounding_notes,
    )
    finalized = validate_and_finalize_ai_submission(
        raw_payload, quant_result, features, calibration_profile=calibration_profile
    )
    return to_plain_dict(finalized)
