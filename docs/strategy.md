# Investment Strategy Documentation

## Overview

The pipeline implements a systematic, rules-based equity selection strategy applied to a cross-sectional universe of stocks. Positions are sized equally, rebalanced monthly, and subject to a set of hard risk constraints enforced at every rebalance.

## Signal Generation

### Factor Universe

The model ingests three categories of predictive signals:

| Category | Examples | Rationale |
|----------|----------|-----------|
| Technical momentum | 3-month, 6-month price return | Trend continuation in equity markets |
| Fundamental quality | Return on equity, earnings yield | Value and profitability premia |
| Risk-adjusted momentum | Volatility-normalised returns | Risk-adjusted trend to reduce whipsaws |

### Cross-Sectional Standardisation

All features are processed cross-sectionally at each rebalance date:
1. **Winsorisation** at 1st / 99th percentile — removes outlier stocks that would otherwise dominate predictions
2. **Z-score normalisation** — converts all signals to comparable scale across the universe
3. **Asof-lag (shift=1)** — ensures only end-of-prior-period values are used, eliminating look-ahead bias

### Prediction Model

An ensemble of XGBoost and Random Forest regressors predicts the next-month cross-sectional return rank. Equal weighting of the two models reduces idiosyncratic overfitting and improves out-of-sample stability.

```
Ensemble score = 0.5 × XGBoost_pred + 0.5 × RandomForest_pred
```

## Portfolio Construction

1. **Stock selection**: Top-N ranked by ensemble score (default: N=20)
2. **Initial weights**: Equal-weight (1/N per holding)
3. **Max individual weight**: capped at 7% after normalisation
4. **Sector concentration**: max 25% per sector
5. **Country concentration**: max 40% per country

## Risk Management

### Beta Hedge Overlay

Portfolio beta is computed vs an equal-weight market proxy using 252 days of daily returns. If β > β_max (1.1 in production), a short market overlay is applied:

```
hedge_ratio = min(0.5, 0.5 × (β - β_max) + 0.1)
```

Net portfolio return is reduced proportionally by the hedge ratio.

### Turnover Cap

Maximum monthly turnover is capped at 30% (production) to limit transaction costs. The constraint is enforced greedily:
1. Maintain existing holdings that remain in the candidate pool
2. Reduce new entries proportionally if total turnover exceeds the cap
3. Preserve residual positions on exits to avoid forced liquidation

### Constraint Monitoring

At every run, the system checks and alerts on:
- `beta_limit`: portfolio beta exceeds β_max
- `sector_limit`: any sector exceeds 25% allocation
- `country_limit`: any country exceeds 40% allocation
- `turnover_limit`: average monthly turnover exceeds the cap

## Transaction Cost Model

| Cost Component | Value |
|----------------|-------|
| Brokerage commission | 3 bps (one-way) |
| Market impact / slippage | 5 bps |
| Bid-ask half-spread | 5 bps |
| Total per-trade cost | 13 bps |
| Management fee | 100 bps/year (prorated monthly) |

Net return per month:
```
ret_net = gross_return - turnover × 13bps - (1 + 100bps)^(1/12) + 1
```

## Backtesting Methodology

Walk-forward validation with expanding training window:

```
Period 1:  [2018-01 ... 2020-01]  → predict 2020-02
Period 2:  [2018-01 ... 2020-02]  → predict 2020-03
...
Period N:  [2018-01 ... 2024-11]  → predict 2024-12
```

Key design choices:
- **Expanding (not rolling) window**: all historical data is used as it becomes available, matching production behaviour
- **1-month signal delay** (`delay_months=1`): signal computed at month-end t is only tradeable at month-end t+1, avoiding implementation shortfall
- **Selection band** (1.5×): candidate pool is widened to 1.5× top_n to allow continuity-first holding (reduces spurious turnover)
