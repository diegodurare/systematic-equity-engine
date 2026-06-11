import numpy as np
import pandas as pd
import pytest

from src.utils.metrics import portfolio_kpis, regression_kpis


def test_portfolio_kpis_empty_returns_nan():
    result = portfolio_kpis(np.array([]))
    assert np.isnan(result["sharpe"])
    assert np.isnan(result["cagr"])
    assert np.isnan(result["max_dd"])


def test_portfolio_kpis_positive_returns_positive_cagr():
    returns = np.full(24, 0.01)  # 1% per month for 2 years
    result = portfolio_kpis(returns)
    assert result["cagr"] > 0


def test_portfolio_kpis_max_dd_is_non_positive():
    rng = np.random.default_rng(99)
    returns = rng.normal(0.005, 0.04, 60)
    result = portfolio_kpis(returns)
    assert result["max_dd"] <= 0


def test_portfolio_kpis_vol_is_annualised():
    monthly_std = 0.02
    returns = np.full(36, monthly_std)
    # vol = std * sqrt(12); for constant series std=0 → but use varied returns
    returns_varied = np.tile([0.02, -0.02], 18)
    result = portfolio_kpis(returns_varied)
    # annualised vol should be std(returns) * sqrt(12)
    expected_vol = float(np.std(returns_varied, ddof=1) * np.sqrt(12))
    assert result["vol"] == pytest.approx(expected_vol, rel=1e-4)


def test_portfolio_kpis_accepts_series():
    s = pd.Series([0.01, 0.02, -0.01, 0.03] * 6)
    result = portfolio_kpis(s)
    assert "sharpe" in result
    assert not np.isnan(result["sharpe"])


def test_regression_kpis_perfect_prediction():
    y = pd.Series([1.0, 2.0, 3.0])
    result = regression_kpis(y, y)
    assert result["mae"] == pytest.approx(0.0)
    assert result["r2"] == pytest.approx(1.0)


def test_regression_kpis_empty_returns_nan():
    result = regression_kpis(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert np.isnan(result["r2"])
