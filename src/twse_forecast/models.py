"""系統共用資料模型與 JSON 序列化工具。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

CollectorStatus = Literal["ok", "degraded", "failed"]


@dataclass(slots=True)
class CollectorReport:
    """單一資料擷取器之執行狀態回報。"""

    source_name: str
    status: CollectorStatus
    fetched_at: str
    message: str
    domains_used: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OHLCVBar:
    """單日還原權息 K 線結構。"""

    date: str  # YYYY-MM-DD
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float


@dataclass(slots=True)
class MarketSeries:
    """單一金融商品歷史序列。"""

    symbol: str
    bars: list[OHLCVBar]

    def closes(self) -> list[float]:
        return [b.close for b in self.bars]

    def adj_closes(self) -> list[float]:
        return [b.adj_close for b in self.bars]

    def dates(self) -> list[str]:
        return [b.date for b in self.bars]


@dataclass(slots=True)
class HorizonQuantForecast:
    """第一層：單一週期純量化基準預測。"""

    horizon_key: str
    label: str
    anchor_trading_days: int
    quant_score: float  # -100 ~ +100
    expected_return: float  # 小數，例如 0.025 代表 +2.5%
    std_dev: float  # 預期報酬率標準差
    prob_up: float
    prob_flat: float
    prob_down: float
    ci80_low_return: float
    ci80_high_return: float
    ci80_low_level: float
    ci80_high_level: float
    key_drivers: dict[str, float]


@dataclass(slots=True)
class HorizonAIForecast:
    """第二層：單一週期 AI 辯證調整後預測（受限於 ±25 分規範）。"""

    horizon_key: str
    ai_score: float
    quant_score: float
    delta_score: float
    expected_return: float
    prob_up: float
    prob_flat: float
    prob_down: float
    ci80_low_level: float
    ci80_high_level: float
    invalidation_level: float
    thesis: str


@dataclass(slots=True)
class AIForecastSubmission:
    """AI 每週提交之完整多空辯證與五週期預測承諾。"""

    model_name: str
    bull_case: str
    bear_case: str
    cio_synthesis: str
    override_rationale: str
    target_allocation: float  # 0.0 ~ 1.0 建議模擬持股水位
    horizons: dict[str, HorizonAIForecast]
    external_news_notes: list[str] = field(default_factory=list)


def to_plain_dict(obj: Any) -> Any:
    """將 dataclass 或巢狀容器安全轉換為標準 JSON 相容字典。"""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain_dict(item) for item in obj]
    return obj
