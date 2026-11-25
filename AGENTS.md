# 台股大盤多週期預測系統 — 代理人執行準則（AGENTS.md）

本工作區為台灣股市大盤（TAIEX `^TWII` 與 `0050.TW`）五週期量化與 AI 辯證預測系統。
凡於本工作區執行預測或績效稽核任務之 AI 代理人，必須嚴格遵守以下規範：

## 1. 語言與數據誠實紀律
- 一律使用**台灣中文**撰寫報告與論述（嚴禁簡體字與中國用語如「板塊、倉位、止損、止盈、市盈率、回撤、信息、質量、周期」）。
- **嚴禁編造或竄改數據**：不得憑空臆測指數點位、籌碼數值或歷史紀錄；一切數字必須來自執行指令後之 `data/snapshots/latest_briefing.md`、`latest_features.json` 與當週即時網路檢索結果。
- **嚴禁修改歷史帳本**：絕對不可編輯或刪除 `data/predictions_ledger.jsonl` 與 `data/ledger.sqlite` 中的過往紀錄。

## 2. 每週一 08:00（Asia/Taipei）標準作業程序
1. **第一階段（確定性資料擷取與基準計算）**：
   執行 `/Users/kevin/.local/bin/uv run --directory "/Users/kevin/Documents/BizIdeas/投資規劃" twse-forecast prepare-weekly`。
2. **第二階段（補充即時新聞與突發風險查核）**：
   讀取 `/Users/kevin/Documents/BizIdeas/投資規劃/data/snapshots/latest_briefing.md`，並使用 `search_web` 補充檢索週末至週一清晨影響台股的國際市場、聯準會官員談話、關稅／地緣政策或台積電與 AI 供應鏈突發新聞。
3. **第三階段（多空辯證與受限調整）**：
   站在「多頭代理人 → 空頭稽核員 → 投資長裁決」三視角交叉辯證，結合上月／近期已結算 Brier Score 誤差檢討，撰寫 `/Users/kevin/Documents/BizIdeas/投資規劃/data/snapshots/latest_ai_forecast.json`。
   - 每個週期（`1_4w`, `1_3m`, `3_6m`, `6_12m`, `1_3y`）之 `ai_score` 與 `quant_score` 差距受限於 `±25` 分內。
   - 明確給出 `invalidation_level`（失效點位）與 `target_allocation`（0.0～1.0 模擬持股水位）。
4. **第四階段（寫入防竄改帳本與產生報表）**：
   執行 `/Users/kevin/.local/bin/uv run --directory "/Users/kevin/Documents/BizIdeas/投資規劃" twse-forecast commit-weekly --forecast-json "/Users/kevin/Documents/BizIdeas/投資規劃/data/snapshots/latest_ai_forecast.json"`。
5. **第五階段（月度績效稽核與完整性驗證）**：
   - 若當日為該月第一個週一（或 `prepare-weekly` 輸出 `MONTHLY_AUDIT_DUE=true`），執行 `/Users/kevin/.local/bin/uv run --directory "/Users/kevin/Documents/BizIdeas/投資規劃" twse-forecast monthly-audit`。
   - 最後必須執行 `/Users/kevin/.local/bin/uv run --directory "/Users/kevin/Documents/BizIdeas/投資規劃" twse-forecast verify-ledger` 確認 `PASS`。
