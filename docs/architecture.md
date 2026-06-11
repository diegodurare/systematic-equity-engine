# Architecture Documentation

## System Overview

The pipeline is organised as a set of decoupled modules under `src/`, orchestrated by `pipeline.py`. Each module has a single responsibility and communicates through well-defined pandas DataFrames and dataclasses.

## Pipeline Architecture

```mermaid
flowchart TB
    subgraph Ingestion["Data Layer (src/data/)"]
        A1[Daily OHLCV CSVs\nprices_por_accion/]
        A2[Factor Scores CSV\nfactores_con_score.csv]
        A3[Fundamentals CSV\nfundamentals_completos.csv]
        A4[Company Metadata\ncompany_metadata.csv]
        A5[Macros\noptional]
    end

    subgraph Features["Feature Engineering (src/features/)"]
        B1[Monthly Return Computation\nmom_3m · mom_6m · vol_3m · vol_6m]
        B2[Data Merge\nprices + static + macros]
        B3[Asof-Lag · Winsorise · Z-score\ncross-sectional standardisation]
    end

    subgraph Models["Prediction Models (src/models/)"]
        C1[XGBoost Regressor\nn_est=400, depth=4, reg_alpha=1]
        C2[Random Forest\nn_est=500, min_leaf=5]
        C3[Ensemble\n0.5 × XGB + 0.5 × RF]
        C4[Walk-Forward Loop\nexpanding window]
    end

    subgraph Portfolio["Portfolio Construction (src/portfolio/)"]
        D1[Top-N Selection\nby ensemble score]
        D2[Equal-Weight\n1/N per holding]
    end

    subgraph Risk["Risk Management (src/risk/)"]
        E1[Weight Limits\nmax 7% per stock]
        E2[Sector / Country Caps\n25% / 40%]
        E3[Beta Overlay\nhedge if β > β_max]
        E4[Turnover Cap\nmax 30%/month]
    end

    subgraph Backtest["Backtesting (src/backtesting/)"]
        F1[Transaction Costs\n3+5+5 bps + 100bps mgmt]
        F2[Selection Band\n1.5× candidate pool]
        F3[Signal Delay\nT+1 month]
    end

    subgraph Output["Metrics & Reporting (src/utils/)"]
        G1[CAGR · Sharpe · Sortino\nMax DD · Calmar · Vol]
        G2[Rolling 36m Metrics]
        G3[Constraint Breach Alerts]
        G4[Run Manifest JSON]
    end

    A1 & A2 & A3 & A4 & A5 --> B1
    B1 --> B2 --> B3
    B3 --> C4
    C4 --> C1 & C2
    C1 & C2 --> C3
    C3 --> D1 --> D2
    D2 --> E1 --> E2 --> E3 --> E4
    E4 --> F1 & F2 & F3
    F1 --> G1 --> G2 --> G3 --> G4
```

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `src/data/loader.py` | Load and type-coerce price/factor/fundamental CSVs |
| `src/features/engineering.py` | Monthly resampling, feature computation, standardisation |
| `src/models/estimators.py` | XGBoost and Random Forest model factory |
| `src/models/walk_forward.py` | Expanding-window train/predict loop |
| `src/portfolio/construction.py` | Top-N selection, equal-weight allocation, benchmark |
| `src/backtesting/engine.py` | Realistic cost backtest with delay and selection band |
| `src/risk/limits.py` | Weight, sector, country, turnover constraints |
| `src/risk/beta.py` | Portfolio beta computation and hedge ratio |
| `src/utils/metrics.py` | Single canonical portfolio KPI implementation |
| `src/utils/monitor.py` | Rolling metrics, breach detection, text reporting |
| `src/utils/artifacts.py` | Artifact persistence, run manifest, directory management |
| `src/utils/config.py` | YAML loading, Paths/Config factory |
| `pipeline.py` | Top-level orchestration, 7-stage run, CLI entry |

## Configuration System

Two config profiles via YAML:

| Parameter | Production | Paper |
|-----------|------------|-------|
| beta_max | 1.1 | 1.2 |
| turnover_cap_m | 0.30 | 0.35 |
| mode | prod | paper |

All other parameters are shared. The pipeline accepts `--config` to switch profiles.

## Data Flow

```
CSV files  →  DataLoader  →  FeatureEngineer  →  WalkForward
                                                       ↓
                                              XGBoost + RF predictions
                                                       ↓
                                              Backtester (top-N selection)
                                                       ↓
                                        Risk constraints (limits, beta)
                                                       ↓
                                        backtest_with_real_costs()
                                                       ↓
                                         portfolio_kpis() + reports
```
