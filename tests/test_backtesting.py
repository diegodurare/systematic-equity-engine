import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import backtest_with_real_costs


def _make_inputs(n: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-31", periods=n, freq="M")
    tickers = [f"T{i}" for i in range(20)]
    rows_pred, rows_ret = [], []
    for d in dates:
        for t in tickers:
            rows_pred.append({"Date": d, "ticker": t, "y_pred": rng.normal(), "return_1m_fwd": 0.01})
            rows_ret.append({"Date": d, "ticker": t, "return_1m": rng.normal(0.005, 0.03)})
    return pd.DataFrame(rows_pred), pd.DataFrame(rows_ret)


def test_backtest_net_return_below_gross():
    """Transaction costs and management fee must reduce net return below gross."""
    preds, returns = _make_inputs(12)
    port = backtest_with_real_costs(preds, returns, top_n=5)
    assert (port["ret"] <= port["gross"]).all()


def test_backtest_first_period_turnover_is_one():
    """On the first holding period there is no prior portfolio → turnover = 1.0."""
    preds, returns = _make_inputs(6)
    port = backtest_with_real_costs(preds, returns, top_n=5, delay_months=0)
    assert port["turnover"].iloc[0] == pytest.approx(1.0)


def test_backtest_turnover_decreases_with_continuity():
    """Subsequent months should have turnover < 1 if holdings overlap."""
    preds, returns = _make_inputs(12)
    port = backtest_with_real_costs(preds, returns, top_n=5)
    assert port["turnover"].iloc[1:].mean() < 1.0


def test_backtest_delay_shifts_signal():
    """With delay_months=1, the first valid row is the second date."""
    preds, returns = _make_inputs(6)
    port_delay = backtest_with_real_costs(preds, returns, top_n=5, delay_months=1)
    port_no_delay = backtest_with_real_costs(preds, returns, top_n=5, delay_months=0)
    assert len(port_delay) == len(port_no_delay) - 1
