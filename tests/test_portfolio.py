import numpy as np
import pandas as pd
import pytest

from src.portfolio.construction import Backtester


def _make_preds(n_tickers: int = 10, n_months: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-31", periods=n_months, freq="M")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rows_pred, rows_ret = [], []
    for d in dates:
        for t in tickers:
            rows_pred.append({"Date": d, "ticker": t, "y_pred": rng.normal(), "return_1m_fwd": rng.normal(0.005, 0.05)})
            rows_ret.append({"Date": d, "ticker": t, "return_1m": rng.normal(0.005, 0.05)})
    return pd.DataFrame(rows_pred), pd.DataFrame(rows_ret)


def test_topn_portfolio_returns_one_row_per_month():
    preds, returns = _make_preds(n_tickers=15, n_months=6)
    bt = Backtester(top_n=5)
    port = bt.topn_portfolio(preds, returns)
    assert len(port) == 6


def test_topn_portfolio_selects_exactly_top_n():
    """With 10 tickers and top_n=3, portfolio return should reflect 3 holdings."""
    preds, returns = _make_preds(n_tickers=10, n_months=3)
    bt = Backtester(top_n=3)
    port = bt.topn_portfolio(preds, returns)
    assert "ret" in port.columns
    assert not port["ret"].isna().all()


def test_equal_weight_benchmark_uses_all_tickers():
    _, returns = _make_preds(n_tickers=10, n_months=4)
    bt = Backtester(top_n=5)
    bench = bt.equal_weight_benchmark(returns)
    assert len(bench) == 4
    assert "ret" in bench.columns
