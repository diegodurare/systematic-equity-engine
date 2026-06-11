import numpy as np
import pandas as pd


class Backtester:
    """Equal-weight top-N portfolio construction from cross-sectional predictions."""

    def __init__(self, top_n: int) -> None:
        self.top_n = top_n

    def topn_portfolio(
        self,
        preds: pd.DataFrame,
        returns_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Select top-N predicted stocks per month and compute equal-weight returns."""
        df = preds.merge(
            returns_df[["Date", "ticker", "return_1m"]],
            on=["Date", "ticker"],
            how="left",
        )
        rows = []
        for date, group in df.groupby("Date"):
            top = group.sort_values("y_pred", ascending=False).head(self.top_n)
            if top.empty:
                continue
            ret = float(np.nansum(top["return_1m"].values * (1.0 / len(top))))
            rows.append({"Date": date, "ret": ret})
        return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)

    def equal_weight_benchmark(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Compute a fully-invested equal-weight benchmark across all available tickers."""
        rows = []
        for date, group in returns_df.groupby("Date"):
            g = group.dropna(subset=["return_1m"])
            if g.empty:
                continue
            rows.append({"Date": date, "ret": float(g["return_1m"].mean())})
        return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
