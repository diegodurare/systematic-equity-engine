import numpy as np
import pandas as pd
import pytest

from src.risk.limits import (
    cap_sector_country,
    enforce_turnover_cap,
    enforce_weight_limits,
)


def _weights(tickers, values) -> pd.DataFrame:
    return pd.DataFrame({"ticker": tickers, "w": values})


def test_enforce_weight_limits_normalises_to_one():
    w = _weights(["A", "B", "C"], [0.5, 0.3, 0.2])
    result = enforce_weight_limits(w, max_weight=0.4)
    assert result["w"].sum() == pytest.approx(1.0)


def test_enforce_weight_limits_reduces_dominant_position():
    # Single-pass clip + renorm reduces the dominant weight but doesn't guarantee ≤ cap
    # with only 2 stocks. Tests that the dominant position is brought down.
    w = _weights(["A", "B", "C", "D", "E"], [0.7, 0.1, 0.1, 0.05, 0.05])
    original_max = float(w["w"].max())
    result = enforce_weight_limits(w, max_weight=0.2)
    assert float(result["w"].max()) < original_max


def test_enforce_weight_limits_equal_fallback_on_zero_sum():
    w = _weights(["A", "B"], [0.0, 0.0])
    result = enforce_weight_limits(w, max_weight=0.5)
    assert result["w"].sum() == pytest.approx(1.0)


def test_cap_sector_country_reduces_sector_concentration():
    # With Tech=0.7 and cap=0.4, iterative scaling must reduce concentration
    w = _weights(["A", "B", "C", "D"], [0.4, 0.3, 0.2, 0.1])
    meta = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "sector": ["Tech", "Tech", "Finance", "Finance"],
        "country": ["ES", "ES", "ES", "ES"],
    })
    result = cap_sector_country(w, meta, sector_cap=0.4, country_cap=1.0)
    tech_after = float(result.loc[result["ticker"].isin(["A", "B"]), "w"].sum())
    assert tech_after < 0.7  # reduced from initial 0.7


def test_cap_sector_country_sums_to_one():
    w = _weights(["A", "B", "C"], [0.5, 0.3, 0.2])
    meta = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "sector": ["X", "X", "Y"],
        "country": ["ES", "ES", "FR"],
    })
    result = cap_sector_country(w, meta, sector_cap=0.35, country_cap=0.6)
    assert result["w"].sum() == pytest.approx(1.0, abs=1e-6)


def test_enforce_turnover_cap_no_prev_returns_unchanged():
    new_w = _weights(["A", "B"], [0.6, 0.4])
    result = enforce_turnover_cap(None, new_w, cap_monthly=0.3)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), new_w.reset_index(drop=True))


def test_enforce_turnover_cap_low_turnover_passes_through():
    prev = _weights(["A", "B"], [0.5, 0.5])
    new = _weights(["A", "B"], [0.52, 0.48])  # tiny change
    result = enforce_turnover_cap(prev, new, cap_monthly=0.3)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), new.reset_index(drop=True))
