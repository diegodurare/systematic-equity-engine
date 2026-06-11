import os

import pandas as pd

from src.utils.metrics import portfolio_kpis


def rolling_metrics(
    port: pd.DataFrame,
    rf_annual: float,
    window_m: int = 36,
) -> pd.DataFrame:
    """Compute rolling Sharpe and max-drawdown over a trailing window."""
    if port.empty:
        return pd.DataFrame(columns=["Date", "sharpe_36m", "maxdd_36m"])

    port = port.sort_values("Date")
    rows = []
    for i in range(window_m, len(port) + 1):
        sl = port.iloc[i - window_m : i]
        k = portfolio_kpis(sl["ret"], rf_annual)
        rows.append({
            "Date": sl["Date"].iloc[-1],
            "sharpe_36m": k["sharpe"],
            "maxdd_36m": k["max_dd"],
        })
    return pd.DataFrame(rows)


def breach_report(
    weights: pd.DataFrame,
    limits: dict,
    exposures: dict,
) -> list[str]:
    """Return a list of constraint names that have been breached."""
    alerts: list[str] = []
    if exposures.get("beta", 0) > limits.get("beta_max", 1.2):
        alerts.append("beta_limit")
    if exposures.get("sector_max", 0) > limits.get("sector_cap", 0.25):
        alerts.append("sector_limit")
    if exposures.get("country_max", 0) > limits.get("country_cap", 0.40):
        alerts.append("country_limit")
    if exposures.get("turnover_m", 0) > limits.get("turnover_cap_m", 0.35):
        alerts.append("turnover_limit")
    return alerts


def render_text_report(
    run_dir: str,
    metrics: dict,
    latest_date: str,
    breaches: list[str],
) -> str:
    """Write a plain-text monthly performance report and return its path."""
    lines = [f"RUN_DIR: {run_dir}", f"LATEST: {latest_date}"]
    for k, v in metrics.items():
        lines.append(f"{k}: {v}")
    lines.append(f"BREACHES: {','.join(breaches) if breaches else 'none'}")

    path = os.path.join(run_dir, "monthly_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
