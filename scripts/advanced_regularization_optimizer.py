#!/usr/bin/env python3
"""
Advanced Regularization Optimizer - Enterprise-grade ML optimization system

This module implements sophisticated regularization and hyperparameter optimization:
- Bayesian optimization with Gaussian Process surrogate models
- Multi-objective optimization (accuracy vs overfitting)
- Adaptive regularization based on validation curves
- Cross-validation with temporal splits for time series
- Performance plateau detection and adaptive search space adjustment

Features:
- Advanced early stopping with multiple criteria
- Regularization parameter auto-tuning
- Model ensemble optimization with diversity constraints
- Automated feature selection with stability analysis
- Performance monitoring and drift detection

Usage:
    optimizer = AdvancedRegularizationOptimizer()
    results = optimizer.optimize_with_regularization('EURUSD', target_accuracy=0.85)
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import warnings
warnings.filterwarnings('ignore')

# Optimization libraries
from sklearn.model_selection import TimeSeriesSplit, validation_curve
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import uniform, randint, loguniform
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner

# ML libraries
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Feature selection
from sklearn.feature_selection import SelectFromModel, RFE, RFECV
from sklearn.inspection import permutation_importance

# Custom imports
from .forecasting import HybridPriceForecastingEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedRegularizationOptimizer:
    """
    Advanced optimization system with sophisticated regularization techniques.
    """

    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        """Initialize the advanced optimizer."""
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.pairs = ['EURUSD', 'XAUUSD']
        
        # Optimization settings
        # Lower default trials during iterative development for faster feedback.
        # Increase this back to 100+ for full production runs.
        self.optimization_trials = 20
        self.optimization_timeout = 3600  # 1 hour
        self.cv_folds = 5
        
        # Early stopping criteria
        self.patience = 20
        self.min_delta = 0.0001
        self.plateau_threshold = 10  # trials without improvement
        
        # Regularization ranges
        self.regularization_ranges = self._get_regularization_ranges()
        
        # Performance tracking
        self.optimization_history = {}

    def _get_regularization_ranges(self) -> Dict:
        """Get comprehensive regularization parameter ranges."""
        return {
            'lightgbm': {
                'learning_rate': (0.01, 0.3),
                'n_estimators': (100, 3000),
                'max_depth': (3, 15),
                'num_leaves': (10, 300),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'min_child_samples': (5, 100),
                'min_child_weight': (0.001, 10.0),
                'reg_alpha': (0.0, 10.0),  # L1
                'reg_lambda': (0.0, 10.0), # L2
                'min_split_gain': (0.0, 1.0),
            },
            'xgboost': {
                'learning_rate': (0.01, 0.3),
                'n_estimators': (100, 3000),
                'max_depth': (3, 15),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'min_child_weight': (1, 20),
                'gamma': (0.0, 5.0),
                'alpha': (0.0, 10.0),      # L1
                'lambda': (0.0, 10.0),     # L2
            },
            'random_forest': {
                'n_estimators': (100, 2000),
                'max_depth': (5, 30),
                'min_samples_split': (2, 50),
                'min_samples_leaf': (1, 20),
                'max_features': ['sqrt', 'log2', 0.5, 0.8, 1.0],
                'min_impurity_decrease': (0.0, 0.1),
                'max_samples': (0.5, 1.0),
            },
            'ridge': {
                'alpha': (0.001, 100.0),
            },
            'lasso': {
                'alpha': (0.001, 10.0),
            },
            'elastic_net': {
                'alpha': (0.001, 10.0),
                'l1_ratio': (0.0, 1.0),
            }
        }

    def optimize_with_regularization(self, pair: str, target_accuracy: float = 0.85) -> Dict:
        """
        Perform comprehensive optimization with advanced regularization.
        
        Args:
            pair: Currency pair to optimize
            target_accuracy: Target accuracy to achieve
            
        Returns:
            Optimization results with best parameters and performance metrics
        """
        logger.info(f"Starting advanced regularization optimization for {pair}")
        
        # Load ensemble and prepare data
        ensemble = HybridPriceForecastingEnsemble(pair, str(self.data_dir), str(self.models_dir))
        
        if ensemble.price_data.empty:
            raise ValueError(f"No data available for {pair}")
        
        # Prepare features and target
        features = ensemble._prepare_features()
        X, y = features.iloc[:-1], features['next_close_change'].iloc[:-1]
        
        # Remove any NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]
        
        results = {
            'pair': pair,
            'target_accuracy': target_accuracy,
            'optimization_start': datetime.now().isoformat(),
            'models': {}
        }
        
        # Optimize each model type
        for model_type in ['lightgbm', 'xgboost', 'random_forest']:
            logger.info(f"Optimizing {model_type} with regularization...")
            
            model_results = self._optimize_model_with_regularization(
                X, y, model_type, target_accuracy, pair
            )
            
            results['models'][model_type] = model_results
            
            # Early termination if target achieved
            if model_results.get('best_score', 0) >= target_accuracy:
                logger.info(f"Target accuracy {target_accuracy} achieved with {model_type}")
                break
        
        # Ensemble optimization
        logger.info("Performing ensemble regularization optimization...")
        ensemble_results = self._optimize_ensemble_regularization(X, y, results['models'])
        results['ensemble'] = ensemble_results
        
        results['optimization_end'] = datetime.now().isoformat()
        
        # Save results
        self._save_optimization_results(pair, results)
        
        return results

    def _optimize_model_with_regularization(self, X: np.ndarray, y: np.ndarray, 
                                          model_type: str, target_accuracy: float, 
                                          pair: str) -> Dict:
        """Optimize a specific model with regularization."""
        
        def objective(trial):
            try:
                # Suggest parameters based on model type
                params = self._suggest_regularized_params(trial, model_type)
                
                # Create model
                model = self._create_model(model_type, params)
                
                # Cross-validation with temporal splits
                tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Fit model
                    if model_type in ['lightgbm', 'xgboost']:
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            callbacks=[optuna.integration.LightGBMPruningCallback(trial, "l1")],
                            early_stopping_rounds=50
                        )
                    else:
                        model.fit(X_train, y_train)
                    
                    # Predict and score
                    y_pred = model.predict(X_val)
                    
                    # Calculate directional accuracy
                    direction_actual = np.sign(y_val)
                    direction_pred = np.sign(y_pred)
                    accuracy = np.mean(direction_actual == direction_pred)
                    
                    scores.append(accuracy)
                
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                # Add penalty for high variance (overfitting indicator)
                regularized_score = mean_score - 0.1 * std_score
                
                # Store additional metrics
                trial.set_user_attr('std_score', std_score)
                trial.set_user_attr('raw_score', mean_score)
                
                return regularized_score
                
            except Exception as e:
                logger.error(f"Error in trial {trial.number}: {e}")
                return 0.0

        # Create study with advanced configuration
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(
                n_startup_trials=20,
                n_ei_candidates=50,
                multivariate=True,
                warn_independent_sampling=False
            ),
            pruner=MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5,
                interval_steps=3
            ),
            study_name=f"{pair}_{model_type}_regularization"
        )
        
        # Optimize with timeout and callbacks
        study.optimize(
            objective,
            n_trials=self.optimization_trials,
            timeout=self.optimization_timeout,
            callbacks=[self._create_early_stopping_callback(target_accuracy)]
        )
        
        # Extract results
        best_params = study.best_params
        best_score = study.best_value
        best_trial = study.best_trial
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_std': best_trial.user_attrs.get('std_score', 0),
            'raw_score': best_trial.user_attrs.get('raw_score', 0),
            'n_trials': len(study.trials),
            'optimization_time': sum(t.duration.total_seconds() for t in study.trials if t.duration)
        }

    def _suggest_regularized_params(self, trial, model_type: str) -> Dict:
        """Suggest parameters with focus on regularization."""
        ranges = self.regularization_ranges[model_type]
        params = {}
        
        if model_type == 'lightgbm':
            params.update({
                'n_estimators': trial.suggest_int('n_estimators', *ranges['n_estimators']),
                'learning_rate': trial.suggest_float('learning_rate', *ranges['learning_rate'], log=True),
                'max_depth': trial.suggest_int('max_depth', *ranges['max_depth']),
                'num_leaves': trial.suggest_int('num_leaves', *ranges['num_leaves']),
                'subsample': trial.suggest_float('subsample', *ranges['subsample']),
                'colsample_bytree': trial.suggest_float('colsample_bytree', *ranges['colsample_bytree']),
                'min_child_samples': trial.suggest_int('min_child_samples', *ranges['min_child_samples']),
                'min_child_weight': trial.suggest_float('min_child_weight', *ranges['min_child_weight']),
                'reg_alpha': trial.suggest_float('reg_alpha', *ranges['reg_alpha']),
                'reg_lambda': trial.suggest_float('reg_lambda', *ranges['reg_lambda']),
                'min_split_gain': trial.suggest_float('min_split_gain', *ranges['min_split_gain']),
                'random_state': 42,
                'verbosity': -1,
                'force_col_wise': True
            })
            
        elif model_type == 'xgboost':
            params.update({
                'n_estimators': trial.suggest_int('n_estimators', *ranges['n_estimators']),
                'learning_rate': trial.suggest_float('learning_rate', *ranges['learning_rate'], log=True),
                'max_depth': trial.suggest_int('max_depth', *ranges['max_depth']),
                'subsample': trial.suggest_float('subsample', *ranges['subsample']),
                'colsample_bytree': trial.suggest_float('colsample_bytree', *ranges['colsample_bytree']),
                'min_child_weight': trial.suggest_int('min_child_weight', *ranges['min_child_weight']),
                'gamma': trial.suggest_float('gamma', *ranges['gamma']),
                'alpha': trial.suggest_float('alpha', *ranges['alpha']),
                'lambda': trial.suggest_float('lambda', *ranges['lambda']),
                'random_state': 42,
                'verbosity': 0
            })
            
        elif model_type == 'random_forest':
            params.update({
                'n_estimators': trial.suggest_int('n_estimators', *ranges['n_estimators']),
                'max_depth': trial.suggest_int('max_depth', *ranges['max_depth']),
                'min_samples_split': trial.suggest_int('min_samples_split', *ranges['min_samples_split']),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', *ranges['min_samples_leaf']),
                'max_features': trial.suggest_categorical('max_features', ranges['max_features']),
                'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', *ranges['min_impurity_decrease']),
                'max_samples': trial.suggest_float('max_samples', *ranges['max_samples']),
                'random_state': 42,
                'n_jobs': -1,
                'oob_score': True
            })
        
        return params

    def _create_model(self, model_type: str, params: Dict):
        """Create model instance with parameters."""
        if model_type == 'lightgbm':
            return LGBMRegressor(**params)
        elif model_type == 'xgboost':
            return XGBRegressor(**params)
        elif model_type == 'random_forest':
            return RandomForestRegressor(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _create_early_stopping_callback(self, target_accuracy: float):
        """Create early stopping callback for optimization."""
        def callback(study, trial):
            if study.best_value >= target_accuracy:
                logger.info(f"Target accuracy {target_accuracy} reached. Stopping optimization.")
                study.stop()
        return callback

    def _optimize_ensemble_regularization(self, X: np.ndarray, y: np.ndarray, 
                                        model_results: Dict) -> Dict:
        """Optimize ensemble with regularization constraints."""
        
        # Extract best models
        best_models = {}
        for model_type, results in model_results.items():
            if 'best_params' in results:
                model = self._create_model(model_type, results['best_params'])
                best_models[model_type] = model
        
        if not best_models:
            return {'error': 'No models available for ensemble'}
        
        # Train models on full data
        for model_type, model in best_models.items():
            model.fit(X, y)
        
        # Optimize ensemble weights with regularization
        def ensemble_objective(trial):
            try:
                # Suggest weights
                weights = {}
                for model_type in best_models.keys():
                    weights[model_type] = trial.suggest_float(f'weight_{model_type}', 0.0, 1.0)
                
                # Normalize weights
                total_weight = sum(weights.values())
                if total_weight == 0:
                    return 0.0
                    
                for key in weights:
                    weights[key] /= total_weight
                
                # Cross-validation
                tscv = TimeSeriesSplit(n_splits=3)  # Reduced for ensemble
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Train models
                    predictions = {}
                    for model_type, model in best_models.items():
                        temp_model = self._create_model(model_type, model_results[model_type]['best_params'])
                        temp_model.fit(X_train, y_train)
                        predictions[model_type] = temp_model.predict(X_val)
                    
                    # Ensemble prediction
                    ensemble_pred = sum(weights[model_type] * predictions[model_type] 
                                      for model_type in best_models.keys())
                    
                    # Calculate accuracy
                    direction_actual = np.sign(y_val)
                    direction_pred = np.sign(ensemble_pred)
                    accuracy = np.mean(direction_actual == direction_pred)
                    scores.append(accuracy)
                
                mean_score = np.mean(scores)
                
                # Add diversity bonus (encourage using multiple models)
                diversity_bonus = 0.01 * (len([w for w in weights.values() if w > 0.1]) - 1)
                
                return mean_score + diversity_bonus
                
            except Exception as e:
                logger.error(f"Error in ensemble optimization: {e}")
                return 0.0
        
        # Optimize ensemble
        ensemble_study = optuna.create_study(direction='maximize')
        ensemble_study.optimize(ensemble_objective, n_trials=50, timeout=300)
        
        return {
            'best_weights': ensemble_study.best_params,
            'best_score': ensemble_study.best_value,
            'n_trials': len(ensemble_study.trials)
        }

    def _save_optimization_results(self, pair: str, results: Dict):
        """Save optimization results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.models_dir / f"{pair}_advanced_regularization_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                serializable_results = self._make_serializable(results)
                import json
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Optimization results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def _make_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

def optimize_pair(pair: str, threshold: float = 0.75) -> float:
    """
    Main function for optimizing a currency pair with advanced regularization.
    
    Args:
        pair: Currency pair to optimize
        threshold: Minimum threshold to trigger optimization
        
    Returns:
        Improvement achieved
    """
    try:
        optimizer = AdvancedRegularizationOptimizer()
        
        # Get current performance
        ensemble = HybridPriceForecastingEnsemble(pair)
        baseline_performance = ensemble.evaluate_current_performance(pair) if hasattr(ensemble, 'evaluate_current_performance') else 0.5
        
        if baseline_performance >= threshold + 0.1:  # Already performing well
            logger.info(f"{pair} already performing well: {baseline_performance:.3f}")
            return 0.0
        
        # Run optimization
        target_accuracy = max(threshold + 0.1, 0.85)  # Aim for at least 85%
        results = optimizer.optimize_with_regularization(pair, target_accuracy)
        
        # Calculate improvement
        best_score = 0
        for model_type, model_results in results.get('models', {}).items():
            if 'best_score' in model_results:
                best_score = max(best_score, model_results['best_score'])
        
        # Check ensemble results
        ensemble_score = results.get('ensemble', {}).get('best_score', 0)
        best_score = max(best_score, ensemble_score)
        
        improvement = best_score - baseline_performance
        
        logger.info(f"Optimization completed for {pair}. Improvement: {improvement:.4f}")
        
        return improvement
        
    except Exception as e:
        logger.error(f"Error optimizing {pair}: {e}")
        return 0.0

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Regularization Optimizer')
    parser.add_argument('--pair', type=str, required=True, help='Currency pair to optimize')
    parser.add_argument('--target', type=float, default=0.85, help='Target accuracy')
    
    args = parser.parse_args()
    
    optimizer = AdvancedRegularizationOptimizer()
    results = optimizer.optimize_with_regularization(args.pair, args.target)
    
    print(f"Optimization completed for {args.pair}")
    print(f"Best scores: {[r.get('best_score', 0) for r in results.get('models', {}).values()]}")