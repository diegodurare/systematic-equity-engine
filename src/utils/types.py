from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Paths:
    factores_path: str
    fundamentals_path: str
    metadata_path: str
    prices_dir: str
    macros_path: Optional[str] = None


@dataclass
class Config:
    lookback_months: int = 24
    top_n: int = 20
    min_training_points: int = 12
    risk_free_annual: float = 0.01
    rebalance_day: str = "M"
