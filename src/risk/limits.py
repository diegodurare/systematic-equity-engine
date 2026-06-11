import numpy as np
import pandas as pd


def _safe_sum(values) -> float:
    try:
        return float(np.nansum(np.asarray(values, dtype=float)))
    except Exception:
        return float(np.nansum(pd.to_numeric(values, errors="coerce").values))


def enforce_weight_limits(weights: pd.DataFrame, max_weight: float) -> pd.DataFrame:
    """Cap individual weights at max_weight and renormalise to sum to 1."""
    w = weights.copy()
    if "w" not in w.columns:
        raise ValueError("weights DataFrame must contain a 'w' column")
    w["w"] = pd.to_numeric(w["w"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if max_weight is not None:
        w["w"] = np.minimum(w["w"], float(max_weight))
    total = _safe_sum(w["w"])
    if total <= 0:
        n = len(w)
        w["w"] = 0.0 if n == 0 else 1.0 / n
        return w
    w["w"] = w["w"] / total
    return w


def cap_sector_country(
    weights: pd.DataFrame,
    meta: pd.DataFrame,
    sector_cap: float,
    country_cap: float,
    iters: int = 10,
) -> pd.DataFrame:
    """
    Iteratively scale sector and country weights to stay within concentration caps.

    Uses a proportional scaling approach: if a group exceeds its cap,
    all members are scaled down proportionally. Runs for `iters` passes
    to handle interactions between overlapping constraints.
    """
    if weights.empty:
        return weights

    w = weights.merge(meta[["ticker", "sector", "country"]], on="ticker", how="left").copy()
    w["w"] = pd.to_numeric(w["w"], errors="coerce").fillna(0.0)
    w["sector"] = w["sector"].fillna("UNK")
    w["country"] = w["country"].fillna("UNK")

    for _ in range(iters):
        if sector_cap is not None:
            for sec, total in w.groupby("sector")["w"].sum().items():
                if total > sector_cap:
                    w.loc[w["sector"] == sec, "w"] *= sector_cap / total

        if country_cap is not None:
            for cty, total in w.groupby("country")["w"].sum().items():
                if total > country_cap:
                    w.loc[w["country"] == cty, "w"] *= country_cap / total

        w["w"] = w["w"].clip(lower=0.0)
        total = _safe_sum(w["w"])
        if total > 0:
            w["w"] /= total

    return w[["ticker", "w"]]


def apply_min_price_liquidity(
    universe_df: pd.DataFrame,
    min_price: float,
    min_adv: float,
    record_reasons: bool = False,
) -> "pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]":
    """Filter universe by minimum price and ADV thresholds."""
    df = universe_df.copy()
    keep = pd.Series(True, index=df.index)
    reasons: list[pd.DataFrame] = []

    if "Close" in df.columns and min_price is not None:
        prices = pd.to_numeric(df["Close"], errors="coerce")
        bad = prices.fillna(np.nan).lt(float(min_price)).fillna(True)
        if record_reasons:
            reasons.append(pd.DataFrame({"ticker": df.loc[bad, "ticker"], "reason": "price"}))
        keep &= ~bad

    if "adv" in df.columns and min_adv is not None:
        adv = pd.to_numeric(df["adv"], errors="coerce").fillna(0.0)
        bad = adv.lt(float(min_adv))
        if record_reasons:
            reasons.append(pd.DataFrame({"ticker": df.loc[bad, "ticker"], "reason": "adv"}))
        keep &= ~bad

    kept = df.loc[keep].copy()
    if record_reasons:
        diag = (
            pd.concat(reasons, ignore_index=True)
            if reasons
            else pd.DataFrame(columns=["ticker", "reason"])
        )
        return kept, diag
    return kept


def enforce_turnover_cap(
    prev_weights: "pd.DataFrame | None",
    new_weights: pd.DataFrame,
    cap_monthly: float,
) -> pd.DataFrame:
    """
    Limit monthly portfolio turnover to cap_monthly.

    Uses a greedy approach: prioritise holding existing positions,
    then reduce new entries proportionally to meet the turnover constraint.
    If no previous weights exist, returns new_weights unchanged.
    """
    if prev_weights is None or prev_weights.empty:
        return new_weights.copy()

    w_prev = prev_weights.set_index("ticker")["w"].astype(float)
    w_new = new_weights.set_index("ticker")["w"].astype(float)

    all_tickers = sorted(set(w_prev.index) | set(w_new.index))
    w_prev = w_prev.reindex(all_tickers).fillna(0.0)
    w_new = w_new.reindex(all_tickers).fillna(0.0)

    if float(np.abs(w_new - w_prev).sum()) <= cap_monthly:
        return new_weights.copy()

    w_adj = w_new.copy()
    entries = (w_prev == 0) & (w_new > 0)
    exits = (w_prev > 0) & (w_new == 0)
    excess = float(np.abs(w_new - w_prev).sum()) - float(cap_monthly)

    # Reduce new entries first
    s_entries = float(w_new[entries].sum())
    if s_entries > 0 and excess > 0:
        to_cut = min(s_entries, excess)
        factor = (s_entries - to_cut) / s_entries
        w_adj[entries] *= factor
        excess -= to_cut

    # Then reduce exits (keep a residual weight to avoid full liquidation)
    if excess > 1e-9:
        exit_idx = np.where(exits.values)[0]
        if len(exit_idx) > 0:
            residual = excess / len(exit_idx)
            w_adj.iloc[exit_idx] = residual

    w_adj = w_adj.clip(lower=0.0)
    total = float(w_adj.sum())
    if total > 0:
        w_adj /= total

    return w_adj.reset_index().rename(columns={0: "w", "index": "ticker"}).assign(
        **{"ticker": w_adj.index, "w": w_adj.values}
    )[["ticker", "w"]]
