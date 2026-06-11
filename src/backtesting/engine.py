import numpy as np
import pandas as pd


def backtest_with_real_costs(
    preds: pd.DataFrame,
    returns_df: pd.DataFrame,
    top_n: int = 20,
    trade_fee_bps: float = 3.0,
    slippage_bps: float = 5.0,
    half_spread_bps: float = 5.0,
    mgmt_fee_annual_bps: float = 100.0,
    selection_band: float = 1.5,
    delay_months: int = 0,
) -> pd.DataFrame:
    """
    Monthly rebalancing backtest with realistic transaction cost modelling.

    Transaction costs are applied per rebalanced unit of turnover:
      - trade_fee:   brokerage commission (one-way)
      - slippage:    market impact on entry/exit
      - half_spread: bid-ask spread cost (half for each side)
      - mgmt_fee:    annual management fee prorated monthly

    selection_band widens the candidate pool above top_n to stabilise holdings
    and reduce unnecessary turnover without changing the final portfolio size.
    delay_months simulates signal-to-execution latency (e.g. T+1 month delay).
    """
    preds = preds.copy()
    preds["Date"] = pd.to_datetime(preds["Date"])
    returns_df = returns_df.copy()
    returns_df["Date"] = pd.to_datetime(returns_df["Date"])

    dates = sorted(preds["Date"].unique().tolist())
    monthly_mgmt = (1 + mgmt_fee_annual_bps / 10_000) ** (1 / 12) - 1
    per_trade_cost = (trade_fee_bps + slippage_bps + half_spread_bps) / 10_000

    # Pre-compute candidate pools with selection band
    candidate_pool: dict = {}
    for date, group in preds.groupby("Date"):
        cutoff = int(np.ceil(top_n * selection_band))
        candidate_pool[date] = (
            group.sort_values("y_pred", ascending=False)
            .head(cutoff)[["ticker", "y_pred"]]
            .reset_index(drop=True)
        )

    rows = []
    prev_holdings: set[str] = set()

    for i, date in enumerate(dates):
        signal_idx = i - delay_months
        if signal_idx < 0:
            continue
        signal_date = dates[signal_idx]
        candidates = candidate_pool[signal_date].sort_values("y_pred", ascending=False)

        # Continuity-first selection: keep existing holdings that remain in candidate pool
        holdings: list[str] = []
        if prev_holdings:
            holdings = [t for t in candidates["ticker"] if t in prev_holdings][:top_n]
        if len(holdings) < top_n:
            new_entries = [t for t in candidates["ticker"] if t not in holdings]
            holdings = (holdings + new_entries)[:top_n]

        month_returns = returns_df[
            (returns_df["Date"] == date) & (returns_df["ticker"].isin(holdings))
        ]
        gross = float(month_returns["return_1m"].mean()) if not month_returns.empty else 0.0

        overlap = len(prev_holdings & set(holdings))
        turnover = 1.0 - overlap / top_n if prev_holdings else 1.0
        tx_cost = turnover * per_trade_cost
        net = gross - tx_cost - monthly_mgmt

        rows.append({
            "Date": date,
            "gross": gross,
            "turnover": turnover,
            "tx_cost": tx_cost,
            "mgmt_fee": monthly_mgmt,
            "ret": net,
        })
        prev_holdings = set(holdings)

    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
