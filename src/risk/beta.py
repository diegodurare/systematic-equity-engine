import numpy as np
import pandas as pd


def market_proxy_from_universe(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Build an equal-weight market proxy from daily prices across the universe."""
    prices = daily_prices.copy()
    prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")
    wide = prices.pivot_table(index="Date", columns="ticker", values="Close", aggfunc="last")
    return wide.mean(axis=1, skipna=True).to_frame("Close_eqw").reset_index()


def compute_beta(
    weights_monthly: pd.DataFrame,
    daily_prices: pd.DataFrame,
    lookback_days: int = 252,
) -> float:
    """
    Compute portfolio beta vs equal-weight market proxy over trailing lookback_days.

    Beta is estimated as cov(r_portfolio, r_market) / var(r_market)
    using daily returns. Portfolio weights are taken from the most recent month.
    """
    if weights_monthly.empty:
        return 0.0

    prices = daily_prices.copy()
    prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")

    last_month = pd.to_datetime(weights_monthly["Date"].max())
    ticker_weights = weights_monthly.groupby("ticker")["w"].last().astype(float)

    sub = prices[prices["ticker"].isin(ticker_weights.index)].copy()
    sub = sub[sub["Date"] <= last_month].sort_values("Date")
    sub = sub.groupby("ticker").tail(lookback_days * 2)  # buffer for gaps
    sub["ret"] = sub.groupby("ticker")["Close"].pct_change()

    wide = (
        sub.pivot_table(index="Date", columns="ticker", values="ret", aggfunc="last")
        .dropna(how="all")
        .tail(lookback_days)
    )
    if wide.empty:
        return 0.0

    w = ticker_weights.reindex(wide.columns).fillna(0.0).values
    ret_port = np.nansum(wide.values * w, axis=1)
    ret_mkt = wide.mean(axis=1, skipna=True).values

    cov = np.nanmean((ret_port - np.nanmean(ret_port)) * (ret_mkt - np.nanmean(ret_mkt)))
    var = np.nanvar(ret_mkt)
    return 0.0 if var == 0 else float(cov / var)


def hedge_ratio_from_beta(
    beta: float,
    beta_max: float,
    max_overlay: float = 0.5,
) -> float:
    """
    Linear hedge overlay: ramp from 0 to max_overlay as beta exceeds beta_max.

    Returns the fraction of portfolio value to hedge via short market exposure.
    """
    if beta <= beta_max:
        return 0.0
    excess = min(beta - beta_max, 1.0)
    return float(min(max_overlay, 0.5 * excess + 0.1))
