## Phase 0: Data Integrity & Preparation

- [x] **Clean Models Directory**
    - Ensure `models/` directory is empty.

- [x] **Audit & Repair Fundamental Data**
    - Run:
      ```bash
      python restore_fundamental_backups.py
      python fix_fundamental_schema.py
      python fix_fundamental_headers.py
      ```
    - Verify all fundamental CSVs have a `date` column and correct value columns.

---

## Sprint 1: Day Trading Signal Engine

- [x] **Signal Engine Implementation**
    - Create `scripts/day_trading_signals.py` with `DayTradingSignalGenerator` class and 10 signal methods.
- [x] **Intraday Feature Engineering**
    - Create `scripts/intraday_features.py` for M15, M30, H1 features.
- [x] **Signal Backtesting**
    - Create `scripts/signal_backtester.py` for intraday backtesting.
- [x] **Integration & Validation**
    - Integrate signals into `forecasting.py` and aggregate to daily level.
    - Unit test all signal methods on sample H1 data.
    - Validate signal columns are present and correct.

---

## Sprint 2: Slump Model Signals

- [x] **Slump Detection Logic**
    - Create `scripts/slump_signals.py` with contrarian slump detection.
- [x] **Pipeline Integration**
    - Integrate slump features into the main pipeline.
- [x] **Testing**
    - Test and validate slump signal generation and win rate features.

---

## Sprint 3: Fundamental & Macro Signals

- [x] **Implement Fundamental Signals**
    - Add 10 fundamental signal types (surprise momentum, yield curve, central bank, volatility jump, etc.).
- [x] **Validation**
    - Ensure all fundamental features are present, non-NaN, and aligned.
- [x] **Integration**
    - Integrate into the master feature matrix.

---

## Sprint 4: Pattern Recognition

- [x] **Candlestick Patterns**
    - Create `scripts/candlestick_patterns.py` and implement 8+ patterns using TA-Lib.
- [x] **Chart & Harmonic Patterns**
    - Create `scripts/chart_patterns.py` and `scripts/harmonic_patterns.py` for classic/harmonic detection.
- [x] **Elliott Wave Patterns**
    - Create `scripts/elliott_wave.py` for wave start signals.
- [x] **Integration & Backtesting**
    - Integrate all pattern signals into the main pipeline.
    - Backtest and validate pattern accuracy.

---

## Final Integration: Ultimate Signal Repository

- [x] **Repository Implementation**
    - Create `scripts/ultimate_signal_repository.py` with `UltimateSignalRepository` class.
- [x] **Module Integration**
    - Integrate all strategy modules (SMC, order flow, scalping, news, carry, harmonic, Elliott, regime, session, etc.).
- [x] **Signal Management**
    - Implement signal ranking, weighting, and master signal generation.
    - Add risk management and performance tracking hooks.
- [x] **Validation**
    - Ensure all signals are present, non-NaN, and trainable together.

---

## Quality Assurance & Validation

- [x] **Diagnostics**
    - Run checks: all features present, no critical NaNs, correct alignment, sufficient variance.
- [x] **Testing**
    - Run unit and integration tests for all signal modules.
- [x] **Backtesting**
    - Validate signal accuracy and win rates.
- [x] **Documentation**
    - Update all `.md` files in `instructions` with implementation status and usage notes.

---

## Ready for Training

- [x] All features and signals are integrated and validated.
- [x] `models/` directory is empty and ready for new training runs.
- [x] Launch full training with the complete, multi-strategy signal set.
- [x] Unit test all signals created.