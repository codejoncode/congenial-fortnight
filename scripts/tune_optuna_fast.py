import optuna
import logging
import time
from scripts.forecasting import HybridPriceForecastingEnsemble

logging.basicConfig(level=logging.INFO)


def objective(trial):
    pair = 'EURUSD'
    ens = HybridPriceForecastingEnsemble(pair)

    # Short settings for speed
    ens.lookback_window = 120

    configs = ens.model_configs

    # Disable heavy/statistical/dl models for fast tuning
    for k in ['prophet', 'auto_arima', 'ets', 'theta', 'lstm', 'bilstm']:
        if k in configs:
            configs[k]['enabled'] = False

    # Sample some hyperparameters
    lgb_n_estimators = trial.suggest_int('lgb_n_estimators', 30, 200)
    lgb_lr = trial.suggest_float('lgb_lr', 0.01, 0.2, log=True)
    lgb_reg_alpha = trial.suggest_float('lgb_reg_alpha', 0.0, 1.0)
    lgb_reg_lambda = trial.suggest_float('lgb_reg_lambda', 0.0, 1.0)

    xgb_n_estimators = trial.suggest_int('xgb_n_estimators', 30, 200)
    xgb_lr = trial.suggest_float('xgb_lr', 0.01, 0.2, log=True)

    rf_n_estimators = trial.suggest_int('rf_n_estimators', 50, 300)

    # Apply to configs
    if 'lightgbm' in configs:
        configs['lightgbm']['params']['n_estimators'] = lgb_n_estimators
        configs['lightgbm']['params']['learning_rate'] = lgb_lr
        configs['lightgbm']['params']['reg_alpha'] = lgb_reg_alpha
        configs['lightgbm']['params']['reg_lambda'] = lgb_reg_lambda

    if 'xgboost' in configs:
        configs['xgboost']['params']['n_estimators'] = xgb_n_estimators
        configs['xgboost']['params']['learning_rate'] = xgb_lr

    if 'random_forest' in configs:
        configs['random_forest']['params']['n_estimators'] = rf_n_estimators

    ens.model_configs = configs

    # Run training (fast) and return directional accuracy
    try:
        metrics = ens.train_full_ensemble()
        # If metric missing, return worst
        da = metrics.get('directional_accuracy', 0.0)
    except Exception as e:
        logging.error(f"Trial failed: {e}")
        da = 0.0

    # Optuna by default minimizes; we return 1 - directional_accuracy to minimize
    return 1.0 - da


def run_study(n_trials: int = 15, timeout: int = 3600):
    start = time.time()
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best = study.best_trial
    print(f"Optuna tuning completed in {time.time()-start:.2f}s")
    print(f"Best trial value (1 - direction_acc): {best.value}")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Also print best directional accuracy
    best_da = 1.0 - best.value
    print(f"Best directional accuracy (approx): {best_da:.4f}")


if __name__ == '__main__':
    run_study(n_trials=15, timeout=1800)
