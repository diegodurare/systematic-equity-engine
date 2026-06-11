import numpy as np
import pandas as pd

# Market-cap thresholds (USD) matching paper Section 4.2
_LARGE_CAP_THRESHOLD = 5_000_000_000   # >5B → large cap
_MID_CAP_THRESHOLD   = 1_000_000_000   # 1-5B → mid cap; <1B → small cap

# Per-trade cost in bps = half-spread + commission (Section 4.2 / Appendix)
_LARGE_BPS = 3.5   # 3 bps spread + 0.5 bps commission
_MID_BPS   = 5.5   # 5 bps spread + 0.5 bps commission
_SMALL_BPS = 5.5   # same as mid (universe floor is 100M USD)

# Flat-rate fallback when no market-cap metadata is provided
_FLAT_BPS  = 13.0  # 3 + 5 + 5 legacy rate


def _avg_cost_bps(
    tickers: list[str],
    ticker_market_cap: dict[str, float] | None,
) -> float:
    """Return the equal-weighted average per-trade cost in bps for `tickers`."""
    if not ticker_market_cap:
        return _FLAT_BPS
    costs = []
    for t in tickers:
        mc = ticker_market_cap.get(t)
        if mc is None:
            costs.append(_MID_BPS)
        elif mc > _LARGE_CAP_THRESHOLD:
            costs.append(_LARGE_BPS)
        elif mc > _MID_CAP_THRESHOLD:
            costs.append(_MID_BPS)
        else:
            costs.append(_SMALL_BPS)
    return float(np.mean(costs)) if costs else _FLAT_BPS


def backtest_with_real_costs(
    preds: pd.DataFrame,
    returns_df: pd.DataFrame,
    top_n: int = 20,
    mgmt_fee_annual_bps: float = 100.0,
    selection_band: float = 1.5,
    delay_months: int = 0,
    ticker_market_cap: dict[str, float] | None = None,
    # Legacy flat-rate parameters kept for backward compatibility;
    # ignored when ticker_market_cap is provided.
    trade_fee_bps: float = 3.0,
    slippage_bps: float = 5.0,
    half_spread_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Monthly rebalancing backtest with tiered transaction cost modelling.

    When `ticker_market_cap` is supplied, per-trade costs are determined by
    market-cap tier (large ≥5B USD: 3.5 bps; mid 1-5B: 5.5 bps; small <1B:
    5.5 bps), matching the cost structure described in Section 4.2 of the paper.
    Without `ticker_market_cap`, the legacy flat rate is used for compatibility.

    Other parameters:
        mgmt_fee_annual_bps: annual management fee prorated monthly.
        selection_band:      widen candidate pool to top_n × band to stabilise
                             holdings and reduce unnecessary turnover.
        delay_months:        signal-to-execution latency in rebalance periods.
    """
    preds = preds.copy()
    preds["Date"] = pd.to_datetime(preds["Date"])
    returns_df = returns_df.copy()
    returns_df["Date"] = pd.to_datetime(returns_df["Date"])

    dates = sorted(preds["Date"].unique().tolist())
    monthly_mgmt = (1 + mgmt_fee_annual_bps / 10_000) ** (1 / 12) - 1

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

        # Continuity-first selection: retain existing holdings still in candidate pool
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

        cost_bps = _avg_cost_bps(holdings, ticker_market_cap)
        tx_cost = turnover * cost_bps / 10_000
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
