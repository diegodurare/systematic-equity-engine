from typing import Optional

import numpy as np
import pandas as pd

from src.data.loader import _coerce_df_numeric


class FeatureEngineer:
    """
    Compute monthly return features from daily prices and merge with static/macro data.

    All features are lagged by one period in transform_asof_standardize() to prevent
    look-ahead bias: at prediction time t, only information confirmed as of t-1 is used.
    """

    def __init__(self, rebalance_day: str = "M") -> None:
        self.rebalance_day = rebalance_day

    def compute_monthly_returns(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Resample daily prices to month-end and compute momentum / volatility features."""
        frames = []
        for ticker, df in prices.items():
            df = df.dropna(subset=["Date", "Close"])
            m = (
                df.set_index("Date")["Close"]
                .resample(self.rebalance_day)
                .last()
                .to_frame("Close")
            )
            m["return_1m"] = m["Close"].pct_change()
            m["return_1m_fwd"] = m["Close"].pct_change().shift(-1)
            m["ret_1m_lag1"] = m["return_1m"].shift(1)
            m["mom_3m"] = m["Close"].pct_change(3)
            m["mom_6m"] = m["Close"].pct_change(6)
            m["vol_3m"] = m["return_1m"].rolling(3).std()
            m["vol_6m"] = m["return_1m"].rolling(6).std()
            m["ticker"] = ticker
            frames.append(m.reset_index())
        return pd.concat(frames, ignore_index=True)

    def merge_all(
        self,
        monthly_prices: pd.DataFrame,
        static_df: pd.DataFrame,
        macros: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Join monthly prices with static factor data and macro indicators."""
        df = monthly_prices.copy()

        if "Date" in static_df.columns:
            s = static_df.copy()
            s["Date"] = pd.to_datetime(s["Date"]).dt.to_period("M").dt.to_timestamp("M")
            df["Date"] = pd.to_datetime(df["Date"]).dt.to_period("M").dt.to_timestamp("M")
            join_cols = ["Date", "ticker"] if "ticker" in s.columns else ["Date"]
            df = df.merge(s, on=join_cols, how="left")

        if macros is not None:
            m = macros.copy()
            m["Date"] = pd.to_datetime(m["Date"]).dt.to_period("M").dt.to_timestamp("M")
            df = df.merge(m, on=["Date"], how="left")

        df = df.sort_values(["Date", "ticker"]).reset_index(drop=True)
        return _coerce_df_numeric(df, exclude=["Date", "ticker"])

    def transform_asof_standardize(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Apply asof-lag, winsorise, and cross-sectional z-score to all numeric features.

        The asof-lag (shift=1 per ticker) ensures that on any prediction date t,
        the model only observes features confirmed at end of period t-1.
        Features with >20% missing values after transformation are dropped.
        """
        df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)
        base_cols = ["Date", "ticker", "return_1m", "return_1m_fwd"]
        num_cols = [
            c for c in df.columns
            if c not in base_cols and df[c].dtype.kind in "fcui"
        ]

        # Asof-lag: forward-fill gaps within each ticker, then shift by 1
        df[num_cols] = df.groupby("ticker")[num_cols].transform(lambda g: g.ffill().shift(1))

        # Cross-sectional winsorise at 1st/99th percentile
        for col in num_cols:
            df[col] = df.groupby("Date")[col].transform(
                lambda s: s.clip(s.quantile(0.01), s.quantile(0.99))
            )

        # Cross-sectional z-score
        for col in num_cols:
            df[col] = df.groupby("Date")[col].transform(
                lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else np.nan)
            )

        keep = [c for c in num_cols if df[c].isna().mean() <= 0.2]
        df = df.dropna(subset=keep + ["return_1m_fwd"]).reset_index(drop=True)
        return df, keep
