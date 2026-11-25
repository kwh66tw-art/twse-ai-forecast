# 台股大盤多週期 AI 預測與月度實證稽核系統（`twse-forecast`）

本專案為針對台灣加權股價指數（TAIEX，`^TWII`）與台灣 50（`0050.TW`）設計的多週期趨勢預測與實證驗證工具，支援 **GitHub Actions 雲端自動化（`gemini-4-neon`）** 與本機指令執行。

## 系統特色

- **五大公開資料管線**：整合 Yahoo Finance 跨市場指標與台股 30 檔代表籃子寬度（含證交所歷史日線備援）、台灣證交所（TWSE）大盤估值／三大法人／融資／月營收、期交所（TAIFEX）期權籌碼、FRED 美債與信用利差、PTT Stock 股板情緒與財經新聞日曆。
- **五週期結構化預測**：涵蓋 `1–4 週`、`1–3 個月`、`3–6 個月`、`6–12 個月`、`1–3 年` 五個時間維度，輸出多／盤／空機率分布、80% 信賴點位區間與失效關鍵點位。
- **雙層架構分離 AI 貢獻**：第一層為確定性量化基準模型；第二層預設透過官方 `google-genai` SDK 呼叫 **`gemini-4-neon`**（開啟 `high` 思考深度、Google Search 即時新聞查核與嚴格 JSON Schema），在 `±25` 分邊界內進行多空辯證調整，以檢驗 AI 相對於純數學公式的淨附加價值（AI Value-Add）。
- **防竄改 SHA-256 雜湊鏈帳本**：同步儲存於 SQLite 與 JSONL，確保歷史預測無法被事後修改。
- **月度實證績效稽核**：自動結算到期預測，計算方向命中率、三分類 Brier Score、80% 信賴區間涵蓋率，以及計入台股 ETF 交易手續費與證交稅後的模擬投資組合淨值與夏普值。

---

## GitHub Actions 雲端自動化部署指南

已內建雲端排程工作流 [`.github/workflows/weekly-forecast.yml`](.github/workflows/weekly-forecast.yml)，設定於 **每週一台灣時間 07:50（UTC 週日 23:50）** 自動執行全流程並寫回儲存庫。

### 1. 取得 Gemini API 金鑰並設定 Repository Secret
1. 前往 Google AI Studio 取得可用的 `GEMINI_API_KEY`（註：`gemini-4-neon` 須於已啟用計費的 Paid Tier 專案中使用）。
2. 在你的 GitHub 私人儲存庫頁面，進入 **Settings → Secrets and variables → Actions → New repository secret**：
   - Name：`GEMINI_API_KEY`
   - Secret：貼上你的 API 金鑰

### 2. 開啟 Actions 自動寫回權限
在 GitHub 儲存庫進入 **Settings → Actions → General → Workflow permissions**：
- 勾選 **Read and write permissions**，點選 **Save**（讓排程可自動提交最新帳本、`reports/` 與 `dashboard/index.html`）。

### 3. 首次推送與手動測試
```bash
git init
git add .
git commit -m "init: TWSE multi-horizon AI forecast system with baseline 2026-W48"
git branch -M main
git remote add origin <你的 GitHub 私人儲存庫 URL>
git push -u origin main
```
推送完成後，可前往 GitHub 的 **Actions → 台股大盤每週多週期預測與月度績效稽核 → Run workflow** 手動測試（支援即時調整 `model`、`thinking_level` 或勾選 `force_monthly_audit`）。

### 4. 模型切換方式
- 預設模型為 `gemini-4-neon`（`thinking_level="high"`）。
- 若想改用較低成本模型或測試不同思考深度，可於手動執行 `workflow_dispatch` 輸入指定模型代碼（例如 `gemini-3.6-flash`），或調整環境變數 `GEMINI_MODEL` 與 `GEMINI_THINKING_LEVEL`。

---

## 本機 CLI 常用指令

```bash
# 步驟 1：執行每週資料擷取、結算到期預測與產生特徵簡報
uv run twse-forecast prepare-weekly

# 步驟 2（雲端／API 模式）：呼叫 gemini-4-neon 產出最新 AI 辯證預測 JSON
GEMINI_API_KEY="你的金鑰" uv run twse-forecast generate-ai

# 步驟 3：寫入 AI 辯證預測至防竄改帳本並產生週報與儀表板
uv run twse-forecast commit-weekly --forecast-json data/snapshots/latest_ai_forecast.json

# 一鍵執行雲端完整流程（含週報、月報判斷與防竄改驗證）
GEMINI_API_KEY="你的金鑰" uv run twse-forecast run-weekly-cloud --skip-if-exists

# 產出月度績效評估報告與更新儀表板
uv run twse-forecast monthly-audit

# 驗證歷史預測帳本雜湊鏈完整性
uv run twse-forecast verify-ledger
```
