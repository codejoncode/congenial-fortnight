from forecasting import HybridPriceForecastingEnsemble
import logging
import time

logging.basicConfig(level=logging.INFO)

def quick_train(pair='EURUSD'):
    start = time.time()
    ens = HybridPriceForecastingEnsemble(pair, data_dir="data", models_dir="models")
    # Reduce lookback and disable heavy modules
    ens.lookback_window = 120  # smaller window
    # Adjust model configs to be light
    configs = ens.model_configs
    # Disable prophet and statsforecast if available for speed
    for k in ['prophet', 'auto_arima', 'ets', 'theta']:
        if k in configs:
            configs[k]['enabled'] = False
    # Reduce ensemble model complexity
    if 'lightgbm' in configs:
        configs['lightgbm']['params']['n_estimators'] = 50
        configs['lightgbm']['params']['learning_rate'] = 0.05
    if 'xgboost' in configs:
        configs['xgboost']['params']['n_estimators'] = 50
        configs['xgboost']['params']['learning_rate'] = 0.05
    # Apply the modified configs
    ens.model_configs = configs

    metrics = ens.train_full_ensemble()
    end = time.time()
    print('Fast training completed in {:.2f}s'.format(end-start))
    print('Metrics:', metrics)

if __name__ == '__main__':
    quick_train()
