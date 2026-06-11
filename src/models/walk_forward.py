import pandas as pd
import numpy as np

from src.models.estimators import build_model
from src.utils.types import Config


class WalkForward:
    """
    Expanding-window walk-forward validation for cross-sectional return prediction.

    At each rebalance date t, the model trains on all available history
    [t - lookback_months, t) and predicts the following month's returns.
    This mirrors how the strategy would operate in production.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def split_dates(
        self, dates: list[pd.Timestamp]
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate (train_start, train_end, test_end) triples for each fold."""
        uniq = sorted(
            pd.to_datetime(pd.Series(dates).dt.to_period("M").dt.to_timestamp("M")).unique()
        )
        return [
            (uniq[i - self.cfg.lookback_months], uniq[i], uniq[i + 1])
            for i in range(self.cfg.lookback_months, len(uniq) - 1)
        ]

    def _prep_xy(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        features: list[str],
        target_col: str,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        x_train = train[features].copy()
        y_train = train[target_col].copy()
        x_test = test[features].copy()

        # Drop constant columns (no information content in training fold)
        const = x_train.columns[x_train.nunique(dropna=True) <= 1].tolist()
        if const:
            x_train = x_train.drop(columns=const)
            x_test = x_test.drop(columns=const)

        med = x_train.median()
        return x_train.fillna(med), y_train, x_test.fillna(med)

    def run(
        self,
        df: pd.DataFrame,
        features: list[str],
        target_col: str,
        model_type: str = "xgb",
    ) -> pd.DataFrame:
        """Execute the full walk-forward loop and return a DataFrame of predictions."""
        results: list[pd.DataFrame] = []

        for train_start, train_end, test_end in self.split_dates(df["Date"].tolist()):
            train = df[(df["Date"] >= train_start) & (df["Date"] < train_end)].dropna(
                subset=[target_col]
            )
            test = df[(df["Date"] >= train_end) & (df["Date"] < test_end)].copy()

            if len(train) < self.cfg.min_training_points or len(test) == 0:
                continue

            x_train, y_train, x_test = self._prep_xy(train, test, features, target_col)
            if x_train.shape[1] == 0:
                continue

            model = build_model(model_type)
            model.fit(x_train, y_train)
            test = test.copy()
            test["y_pred"] = model.predict(x_test)
            results.append(test[["Date", "ticker", target_col, "y_pred"]])

        if not results:
            return pd.DataFrame(columns=["Date", "ticker", target_col, "y_pred"])
        return pd.concat(results, ignore_index=True)

    def run_multiple(
        self,
        df: pd.DataFrame,
        features: list[str],
        target_col: str,
        model_types: list[str],
    ) -> dict[str, pd.DataFrame]:
        """
        Run walk-forward for multiple model types.

        Returns a dict mapping model_type → predictions DataFrame. All predictions
        are out-of-sample by construction (each fold trains on [t-lookback, t) and
        predicts t → t+1). The returned predictions are used by the ensemble
        calibration step to derive optimal blending weights.
        """
        return {mt: self.run(df, features, target_col, mt) for mt in model_types}
