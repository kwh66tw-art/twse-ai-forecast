"""系統核心組態：預測週期規格、台股代表籃子、成本參數與路徑管理。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

TAIPEI_TZ = ZoneInfo("Asia/Taipei")

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 台股模擬部位對標 ETF（0050）真實雙邊交易摩擦成本
ETF_FEE_RATE = 0.001425 * 0.6  # 0.0855% 電子下單 6 折手續費（買進／賣出）
ETF_SELL_TAX_RATE = 0.001  # 0.1% ETF 證券交易稅（僅賣出）

# AI 調整純量化分數之最大允許幅度（確保可量化 AI 淨附加價值）
MAX_AI_SCORE_DELTA = 25.0

# 目標波動率（用於 Barroso & Santa-Clara 波動率管理部位控管）
TARGET_ANNUAL_VOLATILITY = 0.18


@dataclass(frozen=True, slots=True)
class HorizonSpec:
    """單一預測週期的時間錨點與分類門檻規格。"""

    key: str
    label: str
    anchor_trading_days: int
    window_min_days: int
    window_max_days: int
    flat_threshold: float  # 盤整區間半寬（例如 0.015 代表 ±1.5%）
    core_driver: str


HORIZONS: tuple[HorizonSpec, ...] = (
    HorizonSpec(
        key="1_4w",
        label="1–4 週",
        anchor_trading_days=20,
        window_min_days=5,
        window_max_days=20,
        flat_threshold=0.015,
        core_driver="美股夜盤與 ADR 溢價、外資期現貨、P/C Ratio、短期乖離反轉與社群反向情緒",
    ),
    HorizonSpec(
        key="1_3m",
        label="1–3 個月",
        anchor_trading_days=40,
        window_min_days=20,
        window_max_days=60,
        flat_threshold=0.030,
        core_driver="上市月營收年增廣度、三大法人波段買賣超、季線扣抵壓力與事件日曆",
    ),
    HorizonSpec(
        key="3_6m",
        label="3–6 個月",
        anchor_trading_days=90,
        window_min_days=60,
        window_max_days=125,
        flat_threshold=0.050,
        core_driver="3–12 個月價格動能、大戶／籃子市場寬度背離與產業庫存趨勢",
    ),
    HorizonSpec(
        key="6_12m",
        label="6–12 個月",
        anchor_trading_days=180,
        window_min_days=125,
        window_max_days=250,
        flat_threshold=0.080,
        core_driver="中期價格動能、大盤估值分位、美債殖利率曲線與信用利差",
    ),
    HorizonSpec(
        key="1_3y",
        label="1–3 年",
        anchor_trading_days=375,
        window_min_days=250,
        window_max_days=750,
        flat_threshold=0.120,
        core_driver="大盤 P/B 歷史分位數、台股風險溢酬 ERP、貨幣與長期結構週期",
    ),
)

HORIZON_MAP: dict[str, HorizonSpec] = {h.key: h for h in HORIZONS}

# 核心與跨市場指標代號（Yahoo Finance）
CORE_MARKET_SYMBOLS: dict[str, str] = {
    "taiex": "^TWII",
    "etf0050": "0050.TW",
    "tsmc_tw": "2330.TW",
    "tsm_adr": "TSM",
    "sox": "^SOX",
    "nvda": "NVDA",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "sp_fut": "ES=F",
    "vix": "^VIX",
    "us10y": "^TNX",
    "us3m": "^IRX",
    "usd_twd": "TWD=X",
    "dxy": "DX-Y.NYB",
}

# 台股市場寬度代表籃子（30 檔涵蓋半導體、AI 伺服器、電子權值、金融、傳產與航運）
# 用以計算等權重表現與站上均線家數比例，偵測「拉積盤」結構背離
TAIWAN_BREADTH_BASKET: tuple[str, ...] = (
    "2330.TW",
    "2317.TW",
    "2454.TW",
    "2308.TW",
    "2382.TW",
    "3711.TW",
    "2303.TW",
    "3034.TW",
    "3231.TW",
    "6669.TW",
    "2379.TW",
    "3017.TW",
    "2357.TW",
    "2376.TW",
    "3661.TW",
    "2881.TW",
    "2882.TW",
    "2891.TW",
    "2886.TW",
    "2884.TW",
    "2412.TW",
    "1301.TW",
    "1303.TW",
    "2002.TW",
    "1216.TW",
    "2603.TW",
    "2609.TW",
    "2615.TW",
    "2207.TW",
    "9910.TW",
)

# PTT Stock 行為心理學極端詞彙詞典（用作短線反向情緒訊號）
PTT_PANIC_KEYWORDS: tuple[str, ...] = (
    "畢業",
    "斷頭",
    "追繳",
    "睡公園",
    "丸子",
    "崩盤",
    "塊陶",
    "腳麻",
    "跌停",
    "血流成河",
    "人踩人",
    "自殺",
    "跳樓",
    "腰斬",
)

PTT_EUPHORIA_KEYWORDS: tuple[str, ...] = (
    "歐印",
    "all in",
    "信貸",
    "質押",
    "噴出",
    "上看",
    "財富自由",
    "無腦多",
    "送分題",
    "嘎空",
    "漲停",
    "發大財",
    "台積電無敵",
    "買就對了",
)


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """集中管理所有資料、報表與帳本檔案路徑，便於 CLI 與測試隔離使用。"""

    root: Path = DEFAULT_PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"

    @property
    def ledger_sqlite(self) -> Path:
        return self.data_dir / "ledger.sqlite"

    @property
    def ledger_jsonl(self) -> Path:
        return self.data_dir / "predictions_ledger.jsonl"

    @property
    def weekly_reports_dir(self) -> Path:
        return self.root / "reports" / "weekly"

    @property
    def monthly_reports_dir(self) -> Path:
        return self.root / "reports" / "monthly"

    @property
    def dashboard_html(self) -> Path:
        return self.root / "dashboard" / "index.html"

    def ensure_dirs(self) -> None:
        """確保所有必要的輸出目錄均已存在。"""
        for d in (
            self.data_dir,
            self.snapshots_dir,
            self.weekly_reports_dir,
            self.monthly_reports_dir,
            self.dashboard_html.parent,
        ):
            d.mkdir(parents=True, exist_ok=True)
