---
name: twse-weekly-run
description: 每週一 08:00 (Asia/Taipei) 執行台股大盤五大維度資料擷取、AI 多空辯證五週期預測、防竄改帳本鎖定與月度績效評估之標準程序。
---

# 台股每週預測與月度稽核執行手冊

## 觸發時機
每週一台灣時間 08:00 例行排程，或使用者手動要求產生本週大盤預測與月度績效報告時使用。

## 執行步驟

### 步驟 1：執行第一層資料管線與量化基準
```bash
/Users/kevin/.local/bin/uv run --directory "/Users/kevin/Documents/BizIdeas/投資規劃" twse-forecast prepare-weekly
```
確認標準輸出中的基準交易日、加權指數收盤與 `MONTHLY_AUDIT_DUE` 狀態。

### 步驟 2：讀取簡報包並檢索當週突發事件
1. 讀取 `/Users/kevin/Documents/BizIdeas/投資規劃/data/snapshots/latest_briefing.md` 與 `/Users/kevin/Documents/BizIdeas/投資規劃/data/snapshots/latest_quant_baseline.json`。
2. 視需要呼叫 `search_web` 補充最近 48 小時影響台股開盤的關鍵新聞（例如美股電子盤／費半動態、台積電ADR或供應鏈消息、央行／關稅／重大法說）。

### 步驟 3：撰寫 AI 多視角辯證 JSON
將 JSON 寫入 `/Users/kevin/Documents/BizIdeas/投資規劃/data/snapshots/latest_ai_forecast.json`，格式必須嚴格符合：
```json
{
  "model_name": "Gemini 4 Carbon",
  "bull_case": "...",
  "bear_case": "...",
  "cio_synthesis": "...",
  "override_rationale": "...",
  "target_allocation": 0.65,
  "external_news_notes": ["..."],
  "horizons": {
    "1_4w": {"ai_score": 0.0, "invalidation_level": 0.0, "thesis": "..."},
    "1_3m": {"ai_score": 0.0, "invalidation_level": 0.0, "thesis": "..."},
    "3_6m": {"ai_score": 0.0, "invalidation_level": 0.0, "thesis": "..."},
    "6_12m": {"ai_score": 0.0, "invalidation_level": 0.0, "thesis": "..."},
    "1_3y": {"ai_score": 0.0, "invalidation_level": 0.0, "thesis": "..."}
  }
}
```
> 注意：各週期 `ai_score` 相對於 `quant_score` 的差異上限為 `±25` 分；若超過，系統將自動截斷以確保可量化 AI 淨貢獻度。

### 步驟 4：鎖定不可竄改帳本與更新週報／儀表板
```bash
/Users/kevin/.local/bin/uv run --directory "/Users/kevin/Documents/BizIdeas/投資規劃" twse-forecast commit-weekly --forecast-json "/Users/kevin/Documents/BizIdeas/投資規劃/data/snapshots/latest_ai_forecast.json"
```

### 步驟 5：月度評鑑（每月第一個週一）與帳本驗證
若步驟 1 顯示 `MONTHLY_AUDIT_DUE=true`（或當日為每月第一個週一），執行：
```bash
/Users/kevin/.local/bin/uv run --directory "/Users/kevin/Documents/BizIdeas/投資規劃" twse-forecast monthly-audit
```
最後一律執行帳本雜湊鏈驗證：
```bash
/Users/kevin/.local/bin/uv run --directory "/Users/kevin/Documents/BizIdeas/投資規劃" twse-forecast verify-ledger
```
