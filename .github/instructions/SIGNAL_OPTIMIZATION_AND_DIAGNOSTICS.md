# Signal Optimization, Tracking, and Diagnostics Plan

## Goals
- Ensure all signals are clearly named and tracked historically.
- Implement a hyperparameter optimization loop for each signal (and ensemble if enabled).
- Remove signals from optimization once they reach a target or max attempts.
- Report backtested results: earnings %, hit rate, last hit time, by trade type (bull/bear) and pair (EURUSD/XAUUSD).
- Run a full diagnostics pass and check off all implementation boxes.

## Steps

1. **Signal Naming & Historical Tracking**
    - [x] Refactor all signal generation to use unique, descriptive names.
    - [x] Store historical backtest results for each signal, including:
        - Signal name
        - Trade type (bull/bear)
        - Pair (EURUSD/XAUUSD)
        - Earnings %
        - Hit rate
        - Last hit timestamp

2. **Hyperparameter Optimization Loop**
    - [x] For each signal:
        - Run hyperparameter optimization (e.g., grid/random/Optuna search).
        - If signal reaches target metric or max attempts, remove from list.
        - Continue until all signals are optimized or maxed out.
    - [x] Optionally, repeat for ensemble models.

3. **Backtest Reporting**
    - [x] For each signal, output:
        - Percentage earnings by trade type and pair
        - Signal hit frequency
        - Time since last hit

4. **Diagnostics & Validation**
    - [x] Run all unit/integration tests and backtests.
    - [x] Validate that all signals are tracked, named, and reported correctly.
    - [x] Check off all implementation boxes in the project checklist.

## Deliverables
- [x] Refactored signal code with naming and tracking
- [x] Optimization loop script/module
- [x] Backtest reporting script/module
- [x] Updated diagnostics and project checklist
- [x] This plan (.md) in `.github/instructions/`
