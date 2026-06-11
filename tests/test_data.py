import pandas as pd
import pytest

from src.data.loader import _coerce_numeric_series, DataLoader
from src.utils.types import Paths


def test_coerce_removes_percent_sign():
    s = pd.Series(["12.5%", "7.3%"])
    result = _coerce_numeric_series(s)
    assert list(result) == pytest.approx([12.5, 7.3])


def test_coerce_handles_comma_as_decimal():
    s = pd.Series(["1,5", "3,25"])
    result = _coerce_numeric_series(s)
    assert result.iloc[0] == pytest.approx(1.5)
    assert result.iloc[1] == pytest.approx(3.25)


def test_coerce_handles_thousands_separator():
    s = pd.Series(["1,500.0", "20,000.5"])
    result = _coerce_numeric_series(s)
    assert result.iloc[0] == pytest.approx(1500.0)


def test_coerce_returns_nan_for_non_numeric():
    s = pd.Series(["N/A", "n.d.", "—"])
    result = _coerce_numeric_series(s)
    assert result.isna().all()


def test_loader_raises_for_missing_prices_dir(tmp_path):
    paths = Paths(
        factores_path=str(tmp_path / "f.csv"),
        fundamentals_path=str(tmp_path / "fu.csv"),
        metadata_path=str(tmp_path / "m.csv"),
        prices_dir=str(tmp_path / "nonexistent"),
    )
    loader = DataLoader(paths)
    with pytest.raises(FileNotFoundError):
        loader.load_prices()
