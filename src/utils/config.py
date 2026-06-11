import yaml
from pathlib import Path

from src.utils.types import Config, Paths


def load_yaml(path: str | Path) -> dict:
    """Load a YAML configuration file and return as dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def paths_from_config(cfg: dict) -> Paths:
    """Build a Paths dataclass from a pipeline YAML config dict."""
    import os
    base = os.path.abspath(cfg["base_dir"])
    return Paths(
        factores_path=os.path.join(base, cfg["factores"]),
        fundamentals_path=os.path.join(base, cfg["fundamentals"]),
        metadata_path=os.path.join(base, cfg["metadata"]),
        prices_dir=os.path.join(base, cfg["prices_dir"]),
        macros_path=cfg.get("macros") or None,
    )


def pipeline_config_from_yaml(cfg: dict) -> Config:
    """Build a Config dataclass from a pipeline YAML config dict."""
    return Config(
        lookback_months=int(cfg["lookback_months"]),
        top_n=int(cfg["top_n"]),
        risk_free_annual=float(cfg["risk_free_annual"]),
        rebalance_day="M",
    )
