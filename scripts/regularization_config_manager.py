#!/usr/bin/env python3
"""
Enterprise Regularization Configuration Manager

This module manages enterprise-level regularization and optimization configurations:
- Dynamic parameter adjustment based on data characteristics
- Performance-based regularization tuning
- Cross-validation strategy selection
- Early stopping criteria optimization
- Model-specific regularization profiles

Features:
- Automated configuration based on dataset properties
- Performance tracking and adaptive parameter adjustment
- Multi-objective optimization balancing accuracy and generalization
- Regularization strength calibration
- Advanced early stopping with multiple criteria

Usage:
    config_manager = RegularizationConfigManager()
    config = config_manager.get_optimized_config('EURUSD', target_accuracy=0.85)
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RegularizationConfigManager:
    """
    Enterprise-level configuration manager for regularization and optimization.
    """

    def __init__(self):
        """Initialize the configuration manager."""
        self.base_configs = self._get_base_configurations()
        self.performance_profiles = self._load_performance_profiles()
        
    def _get_base_configurations(self) -> Dict:
        """Get base regularization configurations for different scenarios."""
        return {
            'conservative': {
                'description': 'High regularization for stable, generalizable models',
                'lightgbm': {
                    'reg_alpha': (0.5, 2.0),
                    'reg_lambda': (0.5, 2.0),
                    'learning_rate': (0.01, 0.05),
                    'min_child_samples': (50, 200),
                    'subsample': (0.7, 0.85),
                    'colsample_bytree': (0.7, 0.85),
                    'early_stopping_rounds': 150
                },
                'xgboost': {
                    'alpha': (0.5, 2.0),
                    'lambda': (1.0, 3.0),
                    'learning_rate': (0.01, 0.05),
                    'min_child_weight': (5, 15),
                    'gamma': (0.2, 1.0),
                    'subsample': (0.7, 0.85),
                    'early_stopping_rounds': 150
                },
                'random_forest': {
                    'min_samples_split': (20, 50),
                    'min_samples_leaf': (10, 25),
                    'max_features': ['sqrt', 'log2'],
                    'min_impurity_decrease': (0.001, 0.01),
                    'max_samples': (0.7, 0.85)
                }
            },
            'balanced': {
                'description': 'Moderate regularization balancing accuracy and generalization',
                'lightgbm': {
                    'reg_alpha': (0.1, 1.0),
                    'reg_lambda': (0.1, 1.0),
                    'learning_rate': (0.02, 0.08),
                    'min_child_samples': (20, 100),
                    'subsample': (0.75, 0.9),
                    'colsample_bytree': (0.75, 0.9),
                    'early_stopping_rounds': 100
                },
                'xgboost': {
                    'alpha': (0.1, 1.0),
                    'lambda': (0.5, 2.0),
                    'learning_rate': (0.02, 0.08),
                    'min_child_weight': (3, 10),
                    'gamma': (0.1, 0.5),
                    'subsample': (0.75, 0.9),
                    'early_stopping_rounds': 100
                },
                'random_forest': {
                    'min_samples_split': (10, 30),
                    'min_samples_leaf': (4, 15),
                    'max_features': ['sqrt', 'log2', 0.8],
                    'min_impurity_decrease': (0.0001, 0.005),
                    'max_samples': (0.75, 0.9)
                }
            },
            'aggressive': {
                'description': 'Lower regularization for maximum performance on stable data',
                'lightgbm': {
                    'reg_alpha': (0.0, 0.5),
                    'reg_lambda': (0.0, 0.5),
                    'learning_rate': (0.03, 0.1),
                    'min_child_samples': (10, 50),
                    'subsample': (0.8, 0.95),
                    'colsample_bytree': (0.8, 0.95),
                    'early_stopping_rounds': 80
                },
                'xgboost': {
                    'alpha': (0.0, 0.5),
                    'lambda': (0.0, 1.0),
                    'learning_rate': (0.03, 0.1),
                    'min_child_weight': (1, 5),
                    'gamma': (0.0, 0.3),
                    'subsample': (0.8, 0.95),
                    'early_stopping_rounds': 80
                },
                'random_forest': {
                    'min_samples_split': (5, 20),
                    'min_samples_leaf': (2, 10),
                    'max_features': ['sqrt', 'log2', 0.8, 1.0],
                    'min_impurity_decrease': (0.0, 0.002),
                    'max_samples': (0.8, 0.95)
                }
            }
        }

    def _load_performance_profiles(self) -> Dict:
        """Load historical performance profiles for pairs."""
        # In a real implementation, this would load from historical optimization results
        return {
            'EURUSD': {
                'volatility': 'medium',
                'trend_strength': 'moderate',
                'noise_level': 'medium',
                'best_regularization': 'balanced',
                'performance_history': []
            },
            'XAUUSD': {
                'volatility': 'high',
                'trend_strength': 'strong',
                'noise_level': 'high',
                'best_regularization': 'conservative',
                'performance_history': []
            }
        }

    def get_optimized_config(self, pair: str, target_accuracy: float = 0.85,
                           data_characteristics: Optional[Dict] = None) -> Dict:
        """
        Get optimized regularization configuration for a specific pair.
        
        Args:
            pair: Currency pair
            target_accuracy: Target accuracy to achieve
            data_characteristics: Optional data characteristics for tuning
            
        Returns:
            Optimized configuration dictionary
        """
        logger.info(f"Generating optimized configuration for {pair}")
        
        # Analyze data characteristics if not provided
        if data_characteristics is None:
            data_characteristics = self._analyze_data_characteristics(pair)
        
        # Select base configuration strategy
        strategy = self._select_regularization_strategy(pair, data_characteristics, target_accuracy)
        
        # Get base configuration
        base_config = self.base_configs[strategy].copy()
        
        # Customize based on data characteristics
        customized_config = self._customize_for_data_characteristics(
            base_config, data_characteristics, target_accuracy
        )
        
        # Add early stopping and convergence criteria
        customized_config['early_stopping'] = self._get_early_stopping_config(
            pair, target_accuracy, data_characteristics
        )
        
        # Add cross-validation strategy
        customized_config['cross_validation'] = self._get_cv_strategy(
            pair, data_characteristics
        )
        
        # Add optimization settings
        customized_config['optimization'] = self._get_optimization_settings(
            pair, target_accuracy, data_characteristics
        )
        
        customized_config['meta'] = {
            'pair': pair,
            'strategy': strategy,
            'target_accuracy': target_accuracy,
            'generated_at': datetime.now().isoformat(),
            'data_characteristics': data_characteristics
        }
        
        return customized_config

    def _analyze_data_characteristics(self, pair: str) -> Dict:
        """Analyze data characteristics to inform regularization strategy."""
        try:
            # This would load and analyze actual price data
            # For now, use profile-based characteristics
            profile = self.performance_profiles.get(pair, {})
            
            return {
                'volatility': profile.get('volatility', 'medium'),
                'trend_strength': profile.get('trend_strength', 'moderate'),
                'noise_level': profile.get('noise_level', 'medium'),
                'data_quality': 'high',  # Assume high quality
                'sample_size': 'large',  # Assume large sample
                'feature_correlation': 'moderate',
                'stationarity': 'moderate'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing data characteristics for {pair}: {e}")
            return {
                'volatility': 'medium',
                'trend_strength': 'moderate',
                'noise_level': 'medium',
                'data_quality': 'medium',
                'sample_size': 'medium',
                'feature_correlation': 'moderate',
                'stationarity': 'moderate'
            }

    def _select_regularization_strategy(self, pair: str, characteristics: Dict,
                                      target_accuracy: float) -> str:
        """Select the appropriate regularization strategy."""
        
        # High target accuracy with noisy data -> conservative
        if target_accuracy >= 0.9 or characteristics.get('noise_level') == 'high':
            return 'conservative'
        
        # High volatility or poor data quality -> conservative
        if (characteristics.get('volatility') == 'high' or 
            characteristics.get('data_quality') == 'low'):
            return 'conservative'
        
        # Low target accuracy with good data -> aggressive
        if target_accuracy <= 0.75 and characteristics.get('data_quality') == 'high':
            return 'aggressive'
        
        # Default to balanced
        return 'balanced'

    def _customize_for_data_characteristics(self, base_config: Dict,
                                          characteristics: Dict,
                                          target_accuracy: float) -> Dict:
        """Customize configuration based on data characteristics."""
        
        config = base_config.copy()
        
        # Adjust for high volatility
        if characteristics.get('volatility') == 'high':
            for model_type in ['lightgbm', 'xgboost']:
                if model_type in config:
                    # Increase regularization
                    if 'reg_alpha' in config[model_type]:
                        alpha_range = config[model_type]['reg_alpha']
                        config[model_type]['reg_alpha'] = (alpha_range[0] * 1.5, alpha_range[1] * 1.5)
                    if 'reg_lambda' in config[model_type]:
                        lambda_range = config[model_type]['reg_lambda']
                        config[model_type]['reg_lambda'] = (lambda_range[0] * 1.5, lambda_range[1] * 1.5)
        
        # Adjust for high noise
        if characteristics.get('noise_level') == 'high':
            for model_type in config:
                if model_type in ['lightgbm', 'xgboost', 'random_forest']:
                    # Increase minimum samples and decrease learning rate
                    if 'learning_rate' in config[model_type]:
                        lr_range = config[model_type]['learning_rate']
                        config[model_type]['learning_rate'] = (lr_range[0] * 0.7, lr_range[1] * 0.7)
        
        # Adjust for small sample size
        if characteristics.get('sample_size') == 'small':
            for model_type in config:
                if model_type in ['random_forest']:
                    # Reduce complexity
                    config[model_type]['min_samples_split'] = (20, 50)
                    config[model_type]['min_samples_leaf'] = (10, 20)
        
        return config

    def _get_early_stopping_config(self, pair: str, target_accuracy: float,
                                 characteristics: Dict) -> Dict:
        """Get early stopping configuration."""
        
        base_patience = 25
        
        # Adjust patience based on characteristics
        if characteristics.get('volatility') == 'high':
            base_patience = 35  # More patience for volatile data
        elif characteristics.get('noise_level') == 'high':
            base_patience = 40  # Even more patience for noisy data
        
        # Adjust based on target accuracy
        if target_accuracy >= 0.9:
            base_patience = int(base_patience * 1.5)  # More patience for high targets
        
        return {
            'enabled': True,
            'criteria': {
                'convergence': {
                    'patience': base_patience,
                    'min_delta': 0.0001,
                    'monitor': 'validation_score'
                },
                'plateau': {
                    'patience': 15,
                    'min_delta': 0.0005,
                    'monitor': 'validation_score'
                },
                'overfitting': {
                    'patience': 10,
                    'threshold': 0.05,  # Max gap between train and validation
                    'monitor': 'score_gap'
                }
            },
            'target_achievement': {
                'enabled': True,
                'threshold': target_accuracy,
                'confidence_required': 0.95  # Stop if consistently above target
            }
        }

    def _get_cv_strategy(self, pair: str, characteristics: Dict) -> Dict:
        """Get cross-validation strategy."""
        
        base_folds = 5
        
        # Adjust folds based on data characteristics
        if characteristics.get('sample_size') == 'large':
            base_folds = 7
        elif characteristics.get('sample_size') == 'small':
            base_folds = 3
        
        return {
            'type': 'TimeSeriesSplit',
            'n_splits': base_folds,
            'test_size': None,  # Auto-determined
            'gap': 0,  # No gap between train/test
            'validation_strategy': {
                'holdout_ratio': 0.2,
                'rolling_window': True,
                'purged_cv': characteristics.get('volatility') == 'high'
            }
        }

    def _get_optimization_settings(self, pair: str, target_accuracy: float,
                                 characteristics: Dict) -> Dict:
        """Get optimization settings."""
        
        base_trials = 100
        base_timeout = 3600  # 1 hour
        
        # Adjust based on target accuracy
        if target_accuracy >= 0.9:
            base_trials = 200  # More trials for high targets
            base_timeout = 7200  # 2 hours
        elif target_accuracy <= 0.75:
            base_trials = 50   # Fewer trials for lower targets
            base_timeout = 1800  # 30 minutes
        
        return {
            'algorithm': 'TPE',  # Tree-structured Parzen Estimator
            'n_trials': base_trials,
            'timeout': base_timeout,
            'n_startup_trials': max(20, base_trials // 5),
            'multivariate': True,
            'pruning': {
                'enabled': True,
                'algorithm': 'MedianPruner',
                'n_startup_trials': 10,
                'n_warmup_steps': 5,
                'interval_steps': 3
            },
            'parallel_trials': 1,  # Sequential for stability
            'callbacks': {
                'target_achievement': True,
                'progress_tracking': True,
                'performance_monitoring': True
            }
        }

    def update_performance_profile(self, pair: str, optimization_results: Dict):
        """Update performance profile based on optimization results."""
        
        if pair not in self.performance_profiles:
            self.performance_profiles[pair] = {
                'volatility': 'medium',
                'trend_strength': 'moderate', 
                'noise_level': 'medium',
                'best_regularization': 'balanced',
                'performance_history': []
            }
        
        # Extract performance metrics
        performance_entry = {
            'timestamp': datetime.now().isoformat(),
            'target_accuracy': optimization_results.get('target_accuracy', 0.85),
            'achieved_accuracy': self._extract_best_score(optimization_results),
            'strategy_used': optimization_results.get('meta', {}).get('strategy', 'unknown'),
            'trials_completed': sum(
                model_results.get('n_trials', 0) 
                for model_results in optimization_results.get('models', {}).values()
            )
        }
        
        self.performance_profiles[pair]['performance_history'].append(performance_entry)
        
        # Update best regularization strategy based on recent performance
        recent_results = self.performance_profiles[pair]['performance_history'][-5:]  # Last 5 runs
        
        if recent_results:
            best_strategy = max(recent_results, key=lambda x: x['achieved_accuracy'])['strategy_used']
            self.performance_profiles[pair]['best_regularization'] = best_strategy
        
        logger.info(f"Updated performance profile for {pair}")

    def _extract_best_score(self, optimization_results: Dict) -> float:
        """Extract the best score from optimization results."""
        best_score = 0.0
        
        for model_results in optimization_results.get('models', {}).values():
            if 'best_score' in model_results:
                best_score = max(best_score, model_results['best_score'])
        
        ensemble_score = optimization_results.get('ensemble', {}).get('best_score', 0.0)
        best_score = max(best_score, ensemble_score)
        
        return best_score

    def get_adaptive_config(self, pair: str, current_performance: float,
                          target_accuracy: float, iteration: int) -> Dict:
        """Get adaptive configuration based on current performance and iteration."""
        
        # Calculate performance gap
        performance_gap = target_accuracy - current_performance
        
        # Adjust strategy based on performance gap and iteration
        if performance_gap > 0.15:  # Large gap
            if iteration > 20:
                strategy = 'aggressive'  # Try more aggressive approach
            else:
                strategy = 'conservative'  # Start conservative
        elif performance_gap > 0.05:  # Medium gap
            strategy = 'balanced'
        else:  # Small gap
            strategy = 'conservative'  # Fine-tune carefully
        
        # Get base configuration for the selected strategy
        base_config = self.base_configs[strategy].copy()
        
        # Add adaptive adjustments
        adaptive_adjustments = {
            'iteration_based_tuning': {
                'learning_rate_decay': max(0.8, 1.0 - iteration * 0.01),
                'regularization_increase': min(1.5, 1.0 + iteration * 0.02),
                'patience_increase': min(50, 25 + iteration * 2)
            },
            'performance_based_tuning': {
                'aggressive_mode': performance_gap > 0.1,
                'fine_tuning_mode': performance_gap < 0.03,
                'exploration_bonus': max(0, (target_accuracy - current_performance) * 10)
            }
        }
        
        base_config['adaptive_adjustments'] = adaptive_adjustments
        
        return base_config

def get_regularization_config(pair: str, target_accuracy: float = 0.85,
                            current_performance: Optional[float] = None,
                            iteration: Optional[int] = None) -> Dict:
    """
    Main function to get regularization configuration.
    
    Args:
        pair: Currency pair
        target_accuracy: Target accuracy to achieve
        current_performance: Current model performance (for adaptive config)
        iteration: Current iteration (for adaptive config)
        
    Returns:
        Regularization configuration dictionary
    """
    config_manager = RegularizationConfigManager()
    
    if current_performance is not None and iteration is not None:
        # Use adaptive configuration
        return config_manager.get_adaptive_config(pair, current_performance, target_accuracy, iteration)
    else:
        # Use standard optimized configuration
        return config_manager.get_optimized_config(pair, target_accuracy)

if __name__ == '__main__':
    # Example usage
    config_manager = RegularizationConfigManager()
    
    for pair in ['EURUSD', 'XAUUSD']:
        config = config_manager.get_optimized_config(pair, target_accuracy=0.85)
        print(f"\n{pair} Configuration:")
        print(f"Strategy: {config['meta']['strategy']}")
        print(f"Early Stopping Patience: {config['early_stopping']['criteria']['convergence']['patience']}")
        print(f"CV Folds: {config['cross_validation']['n_splits']}")
        print(f"Optimization Trials: {config['optimization']['n_trials']}")