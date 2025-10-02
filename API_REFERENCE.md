# API Reference - Enhanced Regularization System

## Overview
This document provides detailed API reference for the enhanced regularization and early stopping system introduced in version 2.0.0.

---

## 🔧 **Core Classes**

### **AutomatedTrainer**
Enhanced automated training class with advanced early stopping.

```python
class AutomatedTrainer:
    def __init__(self, target_accuracy: float = 0.85, max_iterations: int = 50):
        """
        Initialize enhanced automated trainer.
        
        Args:
            target_accuracy: Target accuracy to achieve (default: 0.85)
            max_iterations: Maximum iterations per pair (default: 50)
        
        Attributes:
            convergence_patience: Iterations without improvement before stopping
            min_improvement: Minimum improvement threshold
            early_stop_threshold: Exceptional performance threshold
            performance_history: Historical performance tracking
            stagnation_counters: Per-pair stagnation detection
        """
```

#### **Methods**

##### **optimize_until_target(pair: str) -> Dict**
```python
def optimize_until_target(self, pair: str) -> Dict:
    """
    Optimize model with advanced early stopping criteria.
    
    Args:
        pair: Currency pair to optimize
        
    Returns:
        Dict containing:
            - pair: Currency pair name
            - final_accuracy: Achieved accuracy
            - iterations_completed: Number of iterations
            - target_reached: Boolean if target achieved
            - results_history: Detailed iteration history
            - stopping_reason: Reason for termination
    
    Early Stopping Criteria:
        1. Target Achievement: >= target_accuracy
        2. Exceptional Performance: >= 0.95
        3. Convergence: No improvement for convergence_patience iterations
        4. Variance Convergence: Recent variance < 0.0001
    """
```

##### **_analyze_performance_trend(recent_performance: List[float]) -> str**
```python
def _analyze_performance_trend(self, recent_performance: List[float]) -> str:
    """
    Analyze recent performance trend.
    
    Args:
        recent_performance: List of recent accuracy values
        
    Returns:
        Trend classification: 'improving', 'declining', 'plateauing', 'insufficient_data'
    """
```

---

## 🎯 **Advanced Regularization Optimizer**

### **AdvancedRegularizationOptimizer**
Bayesian hyperparameter optimization with enterprise-level regularization.

```python
class AdvancedRegularizationOptimizer:
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        """
        Initialize advanced regularization optimizer.
        
        Args:
            data_dir: Directory containing data
            models_dir: Directory for saving models
            
        Attributes:
            optimization_trials: Number of Bayesian optimization trials (100)
            optimization_timeout: Maximum optimization time (3600s)
            cv_folds: Cross-validation folds (5)
            patience: Early stopping patience (20)
        """
```

#### **Methods**

##### **optimize_with_regularization(pair: str, target_accuracy: float) -> Dict**
```python
def optimize_with_regularization(self, pair: str, target_accuracy: float = 0.85) -> Dict:
    """
    Perform comprehensive optimization with advanced regularization.
    
    Args:
        pair: Currency pair to optimize
        target_accuracy: Target accuracy to achieve
        
    Returns:
        Dict containing:
            - pair: Currency pair
            - target_accuracy: Target accuracy
            - models: Results for each model type
                - best_params: Optimized parameters
                - best_score: Achieved score
                - best_std: Score standard deviation
                - n_trials: Number of optimization trials
            - ensemble: Ensemble optimization results
            - optimization_start/end: Timestamps
    
    Features:
        - Bayesian optimization with TPE sampler
        - Multi-fold temporal cross-validation
        - Regularization strength auto-tuning
        - Early stopping with multiple criteria
    """
```

##### **_suggest_regularized_params(trial, model_type: str) -> Dict**
```python
def _suggest_regularized_params(self, trial, model_type: str) -> Dict:
    """
    Suggest regularization parameters for optimization trial.
    
    Args:
        trial: Optuna trial object
        model_type: Type of model ('lightgbm', 'xgboost', 'random_forest')
        
    Returns:
        Dictionary of suggested parameters with regularization focus
        
    Regularization Parameters:
        LightGBM: reg_alpha, reg_lambda, min_child_samples, min_child_weight
        XGBoost: alpha, lambda, min_child_weight, gamma
        Random Forest: min_samples_split, min_samples_leaf, max_features
    """
```

---

## ⚙️ **Configuration Management**

### **RegularizationConfigManager**
Enterprise configuration management for adaptive regularization.

```python
class RegularizationConfigManager:
    def __init__(self):
        """
        Initialize configuration manager.
        
        Strategies:
            - conservative: High regularization for stable models
            - balanced: Moderate regularization for optimal performance  
            - aggressive: Lower regularization for maximum accuracy
        """
```

#### **Methods**

##### **get_optimized_config(pair: str, target_accuracy: float, data_characteristics: Dict) -> Dict**
```python
def get_optimized_config(self, pair: str, target_accuracy: float = 0.85, 
                        data_characteristics: Optional[Dict] = None) -> Dict:
    """
    Get optimized regularization configuration.
    
    Args:
        pair: Currency pair
        target_accuracy: Target accuracy to achieve
        data_characteristics: Optional data characteristics
        
    Returns:
        Configuration dictionary containing:
            - Model-specific regularization parameters
            - Early stopping configuration
            - Cross-validation strategy
            - Optimization settings
            - Meta information
    
    Configuration Selection:
        - High target (>= 0.9) or noisy data -> Conservative
        - High volatility or poor quality -> Conservative
        - Low target (<= 0.75) with good data -> Aggressive
        - Default -> Balanced
    """
```

##### **get_adaptive_config(pair: str, current_performance: float, target_accuracy: float, iteration: int) -> Dict**
```python
def get_adaptive_config(self, pair: str, current_performance: float,
                       target_accuracy: float, iteration: int) -> Dict:
    """
    Get adaptive configuration based on current performance.
    
    Args:
        pair: Currency pair
        current_performance: Current model performance
        target_accuracy: Target accuracy
        iteration: Current iteration number
        
    Returns:
        Adaptive configuration with iteration and performance-based adjustments
        
    Adaptive Logic:
        - Large gap + late iteration -> Aggressive approach
        - Large gap + early iteration -> Conservative approach
        - Medium gap -> Balanced approach
        - Small gap -> Conservative fine-tuning
    """
```

---

## 📊 **Enhanced Model Configurations**

### **Model Parameter Enhancements**

#### **LightGBM Configuration**
```python
'lightgbm': {
    'enabled': True,
    'type': 'ml',
    'params': {
        'n_estimators': 2000,              # Increased from 1000
        'learning_rate': 0.03,             # Reduced from 0.05
        'max_depth': 6,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,           # NEW: Regularization
        'min_child_weight': 0.001,         # NEW: Regularization
        'reg_alpha': 0.1,                  # NEW: L1 regularization
        'reg_lambda': 0.1,                 # NEW: L2 regularization
        'random_state': 42,
        'verbosity': -1,
        'force_col_wise': True,            # NEW: Performance
        'boosting_type': 'gbdt'            # NEW: Explicit boosting
    },
    'early_stopping': {                    # NEW: Early stopping config
        'enabled': True,
        'rounds': 100,
        'metric': 'l2'
    }
}
```

#### **XGBoost Configuration**
```python
'xgboost': {
    'enabled': True,
    'type': 'ml', 
    'params': {
        'n_estimators': 2000,              # Increased from 1000
        'learning_rate': 0.03,             # Reduced from 0.05
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,             # NEW: Regularization
        'gamma': 0.2,                      # NEW: Min loss reduction
        'alpha': 0.1,                      # NEW: L1 regularization
        'lambda': 1.0,                     # NEW: L2 regularization
        'scale_pos_weight': 1,             # NEW: Class balance
        'random_state': 42,
        'verbosity': 0,
        'tree_method': 'auto'              # NEW: Tree method
    },
    'early_stopping': {                    # NEW: Early stopping config
        'enabled': True,
        'rounds': 100,
        'metric': 'rmse'
    }
}
```

#### **Random Forest Configuration**
```python
'random_forest': {
    'enabled': True,
    'type': 'ml',
    'params': {
        'n_estimators': 800,               # Increased from 500
        'max_depth': 12,                   # Increased from 10
        'min_samples_split': 10,           # Increased from 5 (regularization)
        'min_samples_leaf': 4,             # Increased from 2 (regularization)
        'max_features': 'sqrt',            # NEW: Feature regularization
        'min_impurity_decrease': 0.0001,   # NEW: Regularization
        'max_samples': 0.8,                # NEW: Bootstrap regularization
        'random_state': 42,
        'n_jobs': -1,
        'oob_score': True                  # NEW: Out-of-bag scoring
    },
    'validation': {                        # NEW: Validation config
        'use_oob': True,
        'target_score': 0.85
    }
}
```

#### **LSTM Configuration**
```python
'lstm': {
    'enabled': tf is not None,
    'type': 'dl',
    'params': {
        'units': 128,                      # Increased from 64
        'dropout': 0.3,                    # Increased from 0.2
        'recurrent_dropout': 0.2,          # NEW: RNN-specific dropout
        'epochs': 200,                     # Increased from 100
        'batch_size': 64,                  # Increased from 32
        'learning_rate': 0.001,
        'l1_reg': 0.01,                    # NEW: L1 regularization
        'l2_reg': 0.01                     # NEW: L2 regularization
    },
    'early_stopping': {                    # NEW: Advanced early stopping
        'enabled': True,
        'monitor': 'val_loss',
        'patience': 25,                    # Increased from 10
        'min_delta': 0.0001,
        'restore_best_weights': True
    },
    'callbacks': {                         # NEW: Callback system
        'reduce_lr': {
            'enabled': True,
            'monitor': 'val_loss',
            'factor': 0.5,
            'patience': 15,
            'min_lr': 1e-7
        }
    }
}
```

---

## 🔄 **Training Method Enhancements**

### **Enhanced ML Model Training**
```python
def _train_ml_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
    """
    Train ML models with advanced regularization and early stopping.
    
    Enhancements:
        - Validation split for early stopping
        - Model-specific early stopping implementation
        - Performance monitoring and logging
        - OOB score validation for Random Forest
        - Best iteration tracking
    
    Args:
        X_train: Training features
        y_train: Training targets
        
    Returns:
        Dictionary of trained models
    """
```

### **Enhanced Deep Learning Training**
```python
def _train_dl_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
    """
    Train deep learning models with advanced regularization.
    
    Enhancements:
        - Kernel regularization (L1/L2)
        - Advanced dropout (standard + recurrent)
        - Early stopping with patience
        - Learning rate reduction on plateau
        - Validation split increase (0.25)
        - Comprehensive callback system
    
    Args:
        X_train: Training features (will be reshaped for sequences)
        y_train: Training targets
        
    Returns:
        Dictionary of trained models with training history
    """
```

---

## 📈 **Performance Monitoring**

### **Metrics Tracked**
```python
# Performance History Structure
{
    'iteration': int,              # Iteration number
    'accuracy': float,             # Current accuracy
    'improvement': float,          # Improvement from optimization
    'improvement_from_best': float, # Improvement from best so far
    'timestamp': str,              # ISO timestamp
    'stopping_reason': str,        # Reason if stopped early
    'trend': str,                  # Performance trend
    'stagnation_counter': int      # Stagnation counter value
}
```

### **Early Stopping Tracking**
```python
# Early Stopping Reasons
STOPPING_REASONS = {
    'target_achieved': 'Target accuracy reached',
    'exceptional_performance': 'Exceptional performance (>95%)',
    'convergence': 'No improvement for patience iterations', 
    'variance_convergence': 'Performance variance below threshold',
    'max_iterations': 'Maximum iterations reached'
}
```

---

## 🛠️ **Utility Functions**

### **Configuration Helper Functions**
```python
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
```

### **Optimization Helper Functions**
```python
def optimize_pair(pair: str, threshold: float = 0.75) -> float:
    """
    Main function for optimizing a currency pair with advanced regularization.
    
    Args:
        pair: Currency pair to optimize
        threshold: Minimum threshold to trigger optimization
        
    Returns:
        Improvement achieved
        
    Process:
        1. Check current performance vs threshold
        2. Run Bayesian optimization if needed
        3. Apply best parameters
        4. Calculate and return improvement
    """
```

---

## 📊 **Configuration Schemas**

### **Base Configuration Schema**
```python
{
    "meta": {
        "pair": str,                    # Currency pair
        "strategy": str,                # Regularization strategy
        "target_accuracy": float,       # Target accuracy
        "generated_at": str,            # ISO timestamp
        "data_characteristics": dict    # Data analysis results
    },
    "early_stopping": {
        "enabled": bool,
        "criteria": {
            "convergence": {
                "patience": int,
                "min_delta": float,
                "monitor": str
            },
            "plateau": {
                "patience": int, 
                "min_delta": float,
                "monitor": str
            },
            "overfitting": {
                "patience": int,
                "threshold": float,
                "monitor": str
            }
        },
        "target_achievement": {
            "enabled": bool,
            "threshold": float,
            "confidence_required": float
        }
    },
    "cross_validation": {
        "type": str,                    # "TimeSeriesSplit"
        "n_splits": int,
        "test_size": null,
        "gap": int,
        "validation_strategy": {
            "holdout_ratio": float,
            "rolling_window": bool,
            "purged_cv": bool
        }
    },
    "optimization": {
        "algorithm": str,               # "TPE"
        "n_trials": int,
        "timeout": int,
        "n_startup_trials": int,
        "multivariate": bool,
        "pruning": {
            "enabled": bool,
            "algorithm": str,
            "n_startup_trials": int,
            "n_warmup_steps": int,
            "interval_steps": int
        }
    }
}
```

---

## 🔗 **Integration Examples**

### **Basic Integration**
```python
from scripts.automated_training import AutomatedTrainer

# Drop-in replacement - no changes needed
trainer = AutomatedTrainer(target_accuracy=0.85)
results = trainer.run_automated_training(['EURUSD'])
```

### **Advanced Integration**
```python
from scripts.advanced_regularization_optimizer import AdvancedRegularizationOptimizer
from scripts.regularization_config_manager import RegularizationConfigManager

# Advanced usage with custom configuration
config_manager = RegularizationConfigManager()
optimizer = AdvancedRegularizationOptimizer()

# Get adaptive configuration
config = config_manager.get_adaptive_config(
    pair='EURUSD',
    current_performance=0.75,
    target_accuracy=0.85,
    iteration=10
)

# Run optimization with custom config
results = optimizer.optimize_with_regularization('EURUSD', target_accuracy=0.85)
```

### **Callback Integration**
```python
def performance_callback(pair: str, iteration: int, accuracy: float):
    """Custom callback for performance monitoring"""
    print(f"Pair: {pair}, Iteration: {iteration}, Accuracy: {accuracy:.4f}")

# Use with automated trainer
trainer = AutomatedTrainer(target_accuracy=0.85)
trainer.performance_callback = performance_callback
```

---

## 📋 **Error Handling**

### **Common Exceptions**
```python
class RegularizationError(Exception):
    """Base exception for regularization system"""
    pass

class ConfigurationError(RegularizationError):
    """Configuration-related errors"""
    pass

class OptimizationError(RegularizationError):
    """Optimization-related errors"""
    pass

class EarlyStoppingError(RegularizationError):
    """Early stopping-related errors"""
    pass
```

### **Error Recovery**
```python
try:
    from scripts.advanced_regularization_optimizer import optimize_pair
except ImportError:
    # Fallback to basic optimizer
    from scripts.optimizer import optimize_pair
    logger.warning("Using basic optimizer - advanced features unavailable")
```

---

## 🔧 **Debugging & Troubleshooting**

### **Debug Configuration**
```python
# Enable detailed logging
logging.getLogger('scripts.advanced_regularization_optimizer').setLevel(logging.DEBUG)
logging.getLogger('scripts.regularization_config_manager').setLevel(logging.DEBUG)
```

### **Performance Diagnostics**
```python
# Check optimization progress
def diagnose_optimization(results: dict):
    """Diagnose optimization results for debugging"""
    for model_type, model_results in results.get('models', {}).items():
        print(f"{model_type}:")
        print(f"  Best Score: {model_results.get('best_score', 'N/A')}")
        print(f"  Trials: {model_results.get('n_trials', 'N/A')}")
        print(f"  Time: {model_results.get('optimization_time', 'N/A')}s")
```

---

This API reference provides comprehensive documentation for all enhanced features and methods in the regularization system. Use it as a reference for integration and advanced usage scenarios.