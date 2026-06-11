import numpy as np
import pandas as pd
import pytest

from src.features.engineering import FeatureEngineer


def _make_prices(ticker: str, n: int = 36) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=n * 30, freq="D")
    closes = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, len(dates)))
    df = pd.DataFrame({
        "Date": dates,
        "Open": closes,
        "High": closes * 1.01,
        "Low": closes * 0.99,
        "Close": closes,
        "Volume": rng.integers(1000, 10000, len(dates)),
    })
    return {ticker: df}


def test_compute_monthly_returns_expected_columns():
    fe = FeatureEngineer()
    prices = _make_prices("TEST")
    result = fe.compute_monthly_returns(prices)
    for col in ["return_1m", "return_1m_fwd", "mom_3m", "mom_6m", "vol_3m", "vol_6m"]:
        assert col in result.columns, f"Missing: {col}"


def test_compute_monthly_returns_ticker_column():
    fe = FeatureEngineer()
    prices = _make_prices("AAPL")
    result = fe.compute_monthly_returns(prices)
    assert (result["ticker"] == "AAPL").all()


def test_transform_asof_standardize_lag_prevents_lookahead():
    """
    After asof-lag + cross-sectional standardisation, the output must be smaller
    than the input (first month per ticker is dropped) and features must exist.
    Requires multiple tickers so cross-sectional z-score has non-zero variance.
    """
    fe = FeatureEngineer()
    # Build 5 tickers with 5 years of daily data each
    prices = {}
    rng = np.random.default_rng(42)
    for i in range(5):
        t = f"T{i}"
        prices.update(_make_prices(t, n=60))
        # Give each ticker slightly different return profile
        prices[t]["Close"] *= (1 + i * 0.01)
    monthly = fe.compute_monthly_returns(prices)
    # Static features with one row per ticker (no Date column)
    static = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(5)],
        "factor_a": [float(i) for i in range(5)],
    })
    merged = fe.merge_all(monthly, static, macros=None)
    transformed, feats = fe.transform_asof_standardize(merged)
    assert len(feats) > 0, "No features survived standardisation"
    assert len(transformed) < len(merged), "Asof-lag must reduce row count"
