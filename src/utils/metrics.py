"""
Canonical portfolio performance metrics.

Consolidates three previous duplicate implementations:
  core_pipeline.Metrics.portfolio_metrics()
  frictions.eval_portfolio()
  logger_utils.kpis_from_series()
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def portfolio_kpis(
    returns: "np.ndarray | pd.Series",
    rf_annual: float = 0.01,
) -> dict[str, float]:
    """
    Compute risk-adjusted performance metrics from a monthly return series.

    Returns CAGR, Sharpe, Sortino, max drawdown, Calmar ratio, and annualised volatility.
    All annualisation uses sqrt(12) for monthly data.
    """
    if isinstance(returns, pd.Series):
        r = returns.dropna().values
    else:
        r = np.asarray(returns, dtype=float)
        r = r[~np.isnan(r)]

    nan_result: dict[str, float] = dict(
        cagr=np.nan, sharpe=np.nan, sortino=np.nan,
        max_dd=np.nan, calmar=np.nan, vol=np.nan,
    )
    if r.size == 0:
        return nan_result

    rf_m = (1 + rf_annual) ** (1 / 12) - 1
    excess = r - rf_m
    vol = float(np.std(r, ddof=1) * np.sqrt(12))
    std_excess = np.std(excess, ddof=1)
    sharpe = float(np.mean(excess) / std_excess * np.sqrt(12)) if std_excess > 0 else np.nan
    downside = float(np.std(np.minimum(excess, 0), ddof=1) * np.sqrt(12))
    sortino = float(np.mean(excess) / downside) if downside > 0 else np.nan

    equity_curve = np.cumprod(1 + r)
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peaks) / peaks
    max_dd = float(drawdowns.min())

    years = r.size / 12
    cagr = float(equity_curve[-1] ** (1 / years) - 1) if years > 0 else np.nan
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan

    return dict(cagr=cagr, sharpe=sharpe, sortino=sortino, max_dd=max_dd, calmar=calmar, vol=vol)


def regression_kpis(
    y_true: "np.ndarray | pd.Series",
    y_pred: "np.ndarray | pd.Series",
) -> dict[str, float]:
    """Compute regression metrics for return prediction quality."""
    df = pd.DataFrame({"true": y_true, "pred": y_pred}).dropna()
    if df.empty:
        return dict(mse=np.nan, mae=np.nan, r2=np.nan, corr=np.nan)
    return dict(
        mse=float(mean_squared_error(df["true"], df["pred"])),
        mae=float(mean_absolute_error(df["true"], df["pred"])),
        r2=float(r2_score(df["true"], df["pred"])),
        corr=float(np.corrcoef(df["true"], df["pred"])[0, 1]),
    )
