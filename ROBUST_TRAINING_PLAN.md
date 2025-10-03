# Project Enhancement & Robust Training Plan

This document summarizes current status, remaining work before training, and concrete next steps to get a fully gated, production-ready training run.

## 1. Current state (short answers first)

- Pairs actively considered and used in code: EURUSD and XAUUSD (two pairs). The training scripts default to these two unless you pass others.
- The robust loader, Holloway-features, and a safe LightGBM pipeline have been implemented and unit-tested in this branch.
- We have added gating so training will not start unless preflight checks (file presence and minimum rows) pass.

Short answers to your direct questions:

- Have we connected fundamental data to the model alongside timeframes and other features? Partial: The `FundamentalFeatureEngineer` exists and `fundamentals.py` contains fetchers and a unified `fetch_fundamental_features()` API. The integration path into the model pipeline is implemented in `forecasting_integration.py` and `forecasting.py` uses a `FundamentalFeatureEngineer` — the wiring is present, and basic fields are consumed, but you should confirm which fundamental fields are being included in the final feature matrix for each training run (we default to the core fields documented in `tests` fixtures: pe_ratio, ebitda, debt_to_equity).

- Are we using other pair data (cross-pair features)? Yes: `HybridPriceForecastingEnsemble` attempts to load cross-pair data (there's a `cross_pair` concept). The model infrastructure supports using other pairs' data as auxiliary features; the code now loads intraday/monthly/price data for the target pair and attempts to provide cross-pair inputs when available. However, the exact set of cross-pair features used in each experiment depends on the pair selection in `automated_training.py` and the `load_and_prepare_datasets()` implementation.

- Are all data sources being used? We read price timeframes (H1, H4, Daily, Monthly where present), and fundamentals. The robust loader is header-aware and will attempt to fix malformed CSVs. There are still opportunities to better surface which sources are missing at preflight and to log exactly which feature columns make it into each training dataset (I can add a verbosity flag to print a final schema per pair before training).

Summary: we have the plumbing in place — robust loading, feature engineering (including Holloway), fundamentals fetchers, and a safe LightGBM training pipeline — and unit tests have been added and are passing. The remaining work is mostly about final wiring, documentation, and a few configurable choices for how much cross-pair and fundamental data to include in each run.

## 2. What's left before we start a controlled training run (concrete checklist)

Priority tasks (blockers):

- [ ] Confirm the canonical feature set written to disk for each pair before training. (Add logging to `forecasting_integration.py` to emit final feature columns used.)
- [ ] Decide which fundamental fields to include by default (we currently have `pe_ratio`, `ebitda`, `debt_to_equity` in tests). Add mapping in `FundamentalFeatureEngineer` to normalise source-specific names to the canonical names.
- [ ] Decide how to include cross-pair data: 1) append cross-pair price features directly, 2) compute cross-pair derived indicators (correlation, spread, cointegration signals), or 3) use aggregated statistics (rolling correlations/ratios). Implement the chosen method in `forecasting_integration.py` or `HybridPriceForecastingEnsemble._get_cross_pair()`.
- [ ] Add a per-pair summary report printed just before training (rows, columns, NA counts, feature list, fundamental coverage, cross-pair coverage).

Nice-to-have (non-blocking):

- [ ] Persist preflight diagnostics outputs (JSON) to `logs/` for audit/troubleshooting.
- [ ] Add small-sample fallback training configs (already implemented) and log when they were used.
- [ ] Add an optional DB ingestion step (PostgreSQL) — medium-term improvement.

## 3. Recommendations for how to use the cross-pair data now

1) Start simple: include raw lagged returns and rolling realized vol from the single strongest cross-pair (e.g., GBPUSD or USDJPY) alongside the target pair. This is easy and low-risk. Implement in `create_features_for_timeframe()` by joining cross-pair series on timestamp and adding prefixed columns like `GBPUSD_return_lag_1`.

2) Once that's stable, add rolling correlation features between the target pair and each candidate cross-pair (20/50 windows). Use these as regime indicators.

3) If those help, experiment with including more cross-pairs and reduce via PCA/feature selection.

## 4. Running a controlled dry-run training (example)

- Use `scripts/automated_training.py --pairs EURUSD XAUUSD --max-iterations 20 --target 0.80` to run a conservative dry-run. The code will run preflight checks first and abort if critical data is missing.
- To run a single-pair quick test: `python -m scripts.automated_training --pairs EURUSD --max-iterations 10`

## 5. Next steps I will perform (I can do these now if you want me to):

1) Add a per-pair schema report before training (small patch). — I can implement and run it now.
2) Add canonical mapping in `FundamentalFeatureEngineer` to normalise fields from different fundamental providers. — I can implement this if you confirm the canonical keys you want.
3) Wire a simple cross-pair join: add lagged returns and rv for the strongest other pair and include them in the feature matrix. — I can implement and run a dry-run using the current data.

If you'd like me to commit and push these updates, tell me which of the next steps above to implement first. Otherwise I'll start with the per-pair schema report and then wire a simple cross-pair join.