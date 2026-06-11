"""Systematic Investment Pipeline — main entrypoint.

Usage:
    python pipeline.py --config configs/config_prod.yaml
    python pipeline.py --config configs/config_paper.yaml
"""
import argparse
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.backtesting.engine import backtest_with_real_costs
from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer
from src.models.walk_forward import WalkForward
from src.portfolio.construction import Backtester
from src.risk.beta import compute_beta, hedge_ratio_from_beta
from src.risk.limits import (
    apply_min_price_liquidity,
    cap_sector_country,
    enforce_turnover_cap,
    enforce_weight_limits,
)
from src.utils.artifacts import (
    ensure_dir,
    manifest,
    save_artifacts,
    snapshot_run_dir,
    ts_now,
)
from src.utils.config import load_yaml, paths_from_config, pipeline_config_from_yaml
from src.utils.metrics import portfolio_kpis, regression_kpis
from src.utils.monitor import breach_report, render_text_report, rolling_metrics
from src.utils.types import Config, Paths

logger = logging.getLogger(__name__)

_ENSEMBLE_MODELS = ["xgb", "lgbm", "rf", "elasticnet"]
_CALIBRATION_MONTHS = 12


def _log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Core pipeline helpers
# ---------------------------------------------------------------------------

def build_merged(
    paths: Paths, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load, engineer features, and return (raw_merged, standardised, feature_list)."""
    dl = DataLoader(paths)
    fe = FeatureEngineer(cfg.rebalance_day)

    static = dl.load_static()
    prices = dl.load_prices()
    macros = dl.load_macros()

    monthly = fe.compute_monthly_returns(prices)
    merged = fe.merge_all(monthly, static, macros)
    merged_t, feats = fe.transform_asof_standardize(merged)
    return merged, merged_t, feats


def _calibrate_ensemble_weights(
    all_preds: dict[str, pd.DataFrame],
    calibration_months: int = _CALIBRATION_MONTHS,
) -> dict[str, float]:
    """
    Calibrate ensemble weights via non-negative least squares (NNLS) on the
    most recent `calibration_months` months of out-of-sample predictions.

    All walk-forward predictions are inherently out-of-sample (each fold trains
    on [t-lookback, t) and predicts the next period), so the calibration window
    simply selects the most recent observations as the weight-fitting sample,
    matching Section 2.5 of the paper.

    Returns a dict model_type → weight, normalised to sum to 1.
    """
    from scipy.optimize import nnls

    model_types = list(all_preds.keys())

    # Align all models on common (Date, ticker) keys
    merged: pd.DataFrame | None = None
    for mt, df in all_preds.items():
        chunk = df[["Date", "ticker", "return_1m_fwd", "y_pred"]].rename(
            columns={"y_pred": f"pred_{mt}"}
        )
        if merged is None:
            merged = chunk
        else:
            merged = merged.merge(
                chunk[["Date", "ticker", f"pred_{mt}"]],
                on=["Date", "ticker"],
                how="inner",
            )

    if merged is None or merged.empty:
        n = len(model_types)
        return {mt: 1.0 / n for mt in model_types}

    # Use the last `calibration_months` as the calibration window
    max_date = pd.to_datetime(merged["Date"]).max()
    cutoff = max_date - pd.DateOffset(months=calibration_months)
    calib = merged[pd.to_datetime(merged["Date"]) >= cutoff].dropna()

    if len(calib) < max(10, len(model_types)):
        n = len(model_types)
        _log(f"calibration: insufficient data ({len(calib)} rows) — using equal weights")
        return {mt: 1.0 / n for mt in model_types}

    y = calib["return_1m_fwd"].values
    X = calib[[f"pred_{mt}" for mt in model_types]].values

    raw_weights, residual = nnls(X, y)
    total = raw_weights.sum()
    if total < 1e-10:
        n = len(model_types)
        return {mt: 1.0 / n for mt in model_types}

    weights = raw_weights / total
    result = {mt: float(w) for mt, w in zip(model_types, weights)}
    _log(f"calibration: weights={result} (residual={residual:.6f})")
    return result


def fit_predict(paths: Paths, cfg: Config, model_type: str) -> dict:
    """Run walk-forward predictions with a single model type."""
    merged, merged_t, feats = build_merged(paths, cfg)
    df_model = pd.concat(
        [merged_t[["Date", "ticker", "return_1m", "return_1m_fwd"]], merged_t[feats]],
        axis=1,
    )
    preds = WalkForward(cfg).run(df_model, feats, "return_1m_fwd", model_type)
    bt = Backtester(cfg.top_n)
    port = bt.topn_portfolio(preds, merged_t)
    bench = bt.equal_weight_benchmark(merged_t)
    return {
        "merged": merged_t,
        "preds": preds,
        "portfolio": port,
        "benchmark": bench,
        "regression_metrics": regression_kpis(preds["return_1m_fwd"], preds["y_pred"]),
        "portfolio_metrics": portfolio_kpis(port["ret"], cfg.risk_free_annual),
        "benchmark_metrics": portfolio_kpis(bench["ret"], cfg.risk_free_annual),
    }


def fit_predict_ensemble(paths: Paths, cfg: Config) -> dict:
    """
    Run the 4-model heterogeneous ensemble with dynamically calibrated weights.

    Models: XGBoost, LightGBM, Random Forest, ElasticNet (Section 2.5).
    Weights are calibrated monthly via NNLS on the last 12 months of
    out-of-sample predictions, then applied to blend the full prediction set.
    """
    merged, merged_t, feats = build_merged(paths, cfg)
    df_model = pd.concat(
        [merged_t[["Date", "ticker", "return_1m", "return_1m_fwd"]], merged_t[feats]],
        axis=1,
    )

    # Run all models
    wf = WalkForward(cfg)
    all_preds = {}
    for mt in _ENSEMBLE_MODELS:
        _log(f"ensemble: fitting {mt}")
        all_preds[mt] = wf.run(df_model, feats, "return_1m_fwd", mt)

    # Calibrate ensemble weights on last 12 months of OOS predictions
    _log("ensemble: calibrating weights via NNLS")
    w = _calibrate_ensemble_weights(all_preds, calibration_months=_CALIBRATION_MONTHS)

    # Blend predictions using calibrated weights
    base = all_preds["xgb"][["Date", "ticker", "return_1m_fwd"]].copy()
    blended = base.copy()
    blended["y_pred"] = 0.0
    for mt, weight in w.items():
        mt_col = f"pred_{mt}"
        chunk = all_preds[mt][["Date", "ticker", "y_pred"]].rename(columns={"y_pred": mt_col})
        blended = blended.merge(chunk, on=["Date", "ticker"], how="inner")
        blended["y_pred"] += weight * blended[mt_col]

    preds_final = blended[["Date", "ticker", "return_1m_fwd", "y_pred"]]
    returns_df = merged_t[["Date", "ticker", "return_1m"]]

    bt = Backtester(cfg.top_n)
    port = bt.topn_portfolio(preds_final, merged_t)
    bench = bt.equal_weight_benchmark(merged_t)

    return {
        "merged": merged_t,
        "preds": preds_final,
        "portfolio": port,
        "benchmark": bench,
        "portfolio_metrics": portfolio_kpis(port["ret"], cfg.risk_free_annual),
        "benchmark_metrics": portfolio_kpis(bench["ret"], cfg.risk_free_annual),
        "returns_df": returns_df,
        "ensemble_weights": w,
    }


# ---------------------------------------------------------------------------
# Portfolio selection with robustness fallbacks
# ---------------------------------------------------------------------------

def _filter_universe(
    universe: pd.DataFrame,
    min_price: float,
    min_adv: float,
) -> pd.DataFrame:
    result = apply_min_price_liquidity(universe, min_price, min_adv)
    assert isinstance(result, pd.DataFrame)
    return result


def latest_selection_block(
    cfg: dict,
    paths: Paths,
    merged: pd.DataFrame,
    preds: pd.DataFrame,
    run_dir: str,
) -> tuple:
    """
    Identify the most recent rebalance date with sufficient universe depth.

    Applies a 4-level fallback cascade (strict → no ADV → relaxed price → emergency)
    to handle months where liquidity filters reduce the available universe below top_n.
    """
    meta = pd.read_csv(paths.metadata_path)
    dates = sorted(preds["Date"].unique())
    top_n = int(cfg["top_n"])
    min_price = float(cfg.get("min_price", 0.0))
    min_adv = float(cfg.get("min_adv", 0.0))

    chosen_date = chosen_weights = chosen_universe = None

    for date in reversed(dates):
        pm = preds[preds["Date"] == date].copy()
        if pm.empty:
            continue

        uni = (
            merged[merged["Date"] == date][["Date", "ticker", "Close"]]
            .merge(meta, on="ticker", how="left")
        )

        # A: strict filters
        kept = _filter_universe(uni, min_price, min_adv)
        if len(kept) >= top_n:
            chosen_date, chosen_universe = date, kept
            break

        # B: ignore ADV
        kept = _filter_universe(uni, min_price, 0.0)
        if len(kept) >= top_n:
            chosen_date, chosen_universe = date, kept
            break

        # C: relaxed price (50%)
        kept = _filter_universe(uni, min_price * 0.5, 0.0)
        if len(kept) >= top_n:
            chosen_date, chosen_universe = date, kept
            break

        # D: minimal threshold — accept half of top_n
        kept = _filter_universe(uni, min_price * 0.5, 0.0)
        if len(kept) >= max(8, top_n // 2):
            chosen_date, chosen_universe = date, kept
            top_n = max(8, top_n // 2)
            break

    if chosen_date is None:
        last_date = dates[-1]
        cand = preds[preds["Date"] == last_date].sort_values("y_pred", ascending=False).head(20)
        uni = (
            merged[merged["Date"] == last_date][["Date", "ticker", "Close"]]
            .merge(meta, on="ticker", how="left")
        )
        kept = _filter_universe(uni, 0.0, 0.0)
        kept = kept[kept["ticker"].isin(cand["ticker"])]
        if len(kept) < 3:
            raise RuntimeError("Universe too small even after emergency fallback")
        chosen_date, chosen_universe, top_n = last_date, kept, 3

    sel = (
        preds[preds["Date"] == chosen_date]
        .sort_values("y_pred", ascending=False)
        .head(top_n)[["ticker"]]
        .merge(chosen_universe[["ticker"]], on="ticker", how="inner")
    )
    sel = sel.copy()
    sel["w"] = 1.0 / len(sel) if len(sel) > 0 else 0.0
    weights = sel[["ticker", "w"]]

    weights = enforce_weight_limits(weights, cfg["max_weight"])
    weights = cap_sector_country(weights, meta, cfg["sector_cap"], cfg["country_cap"])

    _log(f"selection: date={chosen_date} n={len(weights)}")
    return chosen_date, weights, chosen_universe


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(config_file: str) -> None:
    t0 = time.time()
    _log(f"start config={config_file}")

    cfg = load_yaml(config_file)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(os.path.join(cfg["out_dir"], "phase6", run_id))
    run_dir = snapshot_run_dir(cfg["out_dir"], run_id)

    paths = paths_from_config(cfg)
    core_cfg = pipeline_config_from_yaml(cfg)

    # 1. Ensemble predictions (4 models, dynamically calibrated weights)
    _log("stage=fit_predict_ensemble")
    t1 = time.time()
    res = fit_predict_ensemble(paths, core_cfg)
    _log(f"stage=fit_predict_ensemble done ({int(time.time() - t1)}s) "
         f"weights={res['ensemble_weights']}")

    merged = res["merged"].copy()
    preds = res["preds"].copy()
    returns_df = res["returns_df"].copy()

    date_start = pd.to_datetime(cfg["date_start"]) if cfg.get("date_start") else None
    date_end = pd.to_datetime(cfg["date_end"]) if cfg.get("date_end") else None
    if date_start is not None:
        merged = merged[merged["Date"] >= date_start]
        preds = preds[preds["Date"] >= date_start]
        returns_df = returns_df[returns_df["Date"] >= date_start]
    if date_end is not None:
        merged = merged[merged["Date"] <= date_end]
        preds = preds[preds["Date"] <= date_end]
        returns_df = returns_df[returns_df["Date"] <= date_end]

    # 2. Latest portfolio selection
    _log("stage=latest_selection")
    latest, weights, universe = latest_selection_block(cfg, paths, merged, preds, run_dir)

    # 3. Build ticker → market_cap mapping for tiered cost model
    meta_df = pd.read_csv(paths.metadata_path)
    ticker_market_cap: dict[str, float] | None = None
    if "market_cap_usd" in meta_df.columns:
        ticker_market_cap = dict(
            zip(meta_df["ticker"], meta_df["market_cap_usd"].astype(float))
        )
        _log(f"cost_model: tiered (market_cap available for {len(ticker_market_cap)} tickers)")
    else:
        _log("cost_model: flat 13bps (market_cap_usd column not in metadata)")

    # 4. Beta hedge overlay
    _log("stage=risk_overlay")
    daily_candidates = []
    for fname in os.listdir(paths.prices_dir):
        if not fname.lower().endswith(".csv"):
            continue
        try:
            d = pd.read_csv(os.path.join(paths.prices_dir, fname), parse_dates=["Date"])
            if {"Date", "ticker", "Close"}.issubset(d.columns):
                daily_candidates.append(d[["Date", "ticker", "Close"]])
        except Exception:
            pass

    beta = hedge_ratio = 0.0
    if daily_candidates:
        daily_prices = pd.concat(daily_candidates, ignore_index=True)
        w_month = weights.copy()
        w_month["Date"] = pd.to_datetime(latest)
        beta = compute_beta(w_month[["Date", "ticker", "w"]], daily_prices, lookback_days=252)
        hedge_ratio = hedge_ratio_from_beta(beta, float(cfg["beta_max"]), max_overlay=0.5)
    _log(f"risk_overlay: beta={beta:.3f} hedge_ratio={hedge_ratio:.3f}")

    # 5. Turnover cap
    _log("stage=turnover_cap")
    last_weights_path = os.path.join(cfg["out_dir"], "phase6", "last_weights.csv")
    prev_weights = None
    if os.path.exists(last_weights_path):
        pw = pd.read_csv(last_weights_path)
        if {"ticker", "w"}.issubset(pw.columns):
            prev_weights = pw

    weights_final = enforce_turnover_cap(prev_weights, weights, float(cfg["turnover_cap_m"]))
    pd.DataFrame(weights_final).to_csv(last_weights_path, index=False)

    # 6. Backtest with tiered transaction costs
    _log("stage=backtest")
    t3 = time.time()
    port = backtest_with_real_costs(
        preds=preds,
        returns_df=returns_df,
        top_n=int(cfg["top_n"]),
        mgmt_fee_annual_bps=100.0,
        selection_band=float(cfg["selection_band"]),
        delay_months=int(cfg["delay_months"]),
        ticker_market_cap=ticker_market_cap,
    )
    if hedge_ratio > 0 and "ret" in port.columns:
        port["ret"] = port["ret"] * (1.0 - hedge_ratio)
    if port.empty:
        raise RuntimeError("Empty portfolio — check data coverage")
    _log(f"stage=backtest done ({int(time.time() - t3)}s) rows={len(port)}")

    # 7. Metrics and alerts
    _log("stage=metrics")
    mets = portfolio_kpis(port["ret"], float(cfg["risk_free_annual"]))
    roll = rolling_metrics(port, rf_annual=float(cfg["risk_free_annual"]), window_m=36)

    x = weights_final.merge(meta_df[["ticker", "sector", "country"]], on="ticker", how="left")
    sec_max = float(x.groupby("sector")["w"].sum().max()) if not x.empty else 0.0
    cty_max = float(x.groupby("country")["w"].sum().max()) if not x.empty else 0.0

    alerts = breach_report(
        weights_final,
        {"beta_max": float(cfg["beta_max"]), "sector_cap": float(cfg["sector_cap"]),
         "country_cap": float(cfg["country_cap"]), "turnover_cap_m": float(cfg["turnover_cap_m"])},
        {"beta": beta, "sector_max": sec_max, "country_max": cty_max,
         "turnover_m": float(port["turnover"].mean()) if "turnover" in port.columns else 0.0},
    )
    _log(f"metrics: cagr={mets.get('cagr', float('nan')):.4f} "
         f"sharpe={mets.get('sharpe', float('nan')):.3f} "
         f"max_dd={mets.get('max_dd', float('nan')):.3f}")

    # 8. Persist outputs
    _log("stage=save_artifacts")
    artifacts = save_artifacts(run_dir, {
        "preds": preds, "weights": weights_final,
        "portfolio": port, "rolling_36m": roll,
    })
    render_text_report(run_dir, mets, str(pd.to_datetime(latest).date()), alerts)
    manifest(run_dir, {
        "generated_at": ts_now(), "run_id": run_id, "config": config_file,
        "latest": str(pd.to_datetime(latest).date()),
        "metrics": mets, "artifacts": artifacts, "alerts": alerts,
        "ensemble_weights": res["ensemble_weights"],
        "elapsed_sec": int(time.time() - t0),
    })
    _log(f"done ({int(time.time() - t0)}s) — outputs in {run_dir}")
    print(mets)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Systematic Investment Pipeline")
    ap.add_argument("--config", default="configs/config_prod.yaml", help="YAML config file")
    args = ap.parse_args()
    main(args.config)
