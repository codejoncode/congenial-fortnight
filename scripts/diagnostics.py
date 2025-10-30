#!/usr/bin/env python3
"""
ModelDiagnosticsFramework - Comprehensive ML model evaluation and diagnostics

This module provides enterprise-grade model diagnostics including:
- Performance metrics and stability analysis
- Feature importance and contribution analysis
- Error analysis and bias detection
- Model comparison and validation
- Automated improvement recommendations
- Cross-validation and robustness testing

Usage:
    # Analyze model performance
    diagnostics = ModelDiagnosticsFramework('EURUSD')
    report = diagnostics.generate_full_report()

    # Get improvement recommendations
    recommendations = diagnostics.get_improvement_recommendations()
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML and stats libraries
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_percentage_error
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Try to import optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from yellowbrick.model_selection import LearningCurve
    YELLOWBRICK_AVAILABLE = True
except ImportError:
    YELLOWBRICK_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelDiagnosticsFramework:
    """
    Comprehensive framework for ML model diagnostics and evaluation.

    Provides detailed analysis of model performance, stability, and areas for improvement.
    """

    def __init__(self, pair: str, data_dir: str = "data", models_dir: str = "models", output_dir: str = "diagnostics"):
        """
        Initialize the diagnostics framework.

        Args:
            pair: Currency pair (e.g., 'EURUSD', 'XAUUSD')
            data_dir: Directory containing data
            models_dir: Directory containing trained models
            output_dir: Directory to save diagnostic outputs
        """
        self.pair = pair
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load models and data
        self.models = {}
        self.price_data = pd.DataFrame()
        self.feature_data = pd.DataFrame()
        self.predictions = {}
        self.actuals = {}

        self._load_models()
        self._load_data()

        # Diagnostic results
        self.diagnostic_report = {}

    def _load_models(self):
        """Load trained models for the currency pair."""
        try:
            # Load ensemble model
            ensemble_file = self.models_dir / f"{self.pair}_ensemble.joblib"
            if ensemble_file.exists():
                ensemble_data = joblib.load(ensemble_file)
                self.models['ensemble'] = ensemble_data
                logger.info(f"Loaded ensemble model for {self.pair}")
            else:
                logger.warning(f"Ensemble model not found: {ensemble_file}")

            # Load individual models
            model_files = [
                f"{self.pair}_rf.joblib",
                f"{self.pair}_xgb.joblib",
                f"{self.pair}_calibrator.joblib",
                f"{self.pair}_scaler.joblib"
            ]

            for model_file in model_files:
                file_path = self.models_dir / model_file
                if file_path.exists():
                    model_name = model_file.replace(f"{self.pair}_", "").replace(".joblib", "")
                    self.models[model_name] = joblib.load(file_path)
                    logger.info(f"Loaded {model_name} model")

        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def _load_data(self):
        """Load price and feature data for analysis."""
        try:
            # Load price data
            price_file = self.data_dir / "raw" / f"{self.pair}_Daily.csv"
            if price_file.exists():
                self.price_data = pd.read_csv(price_file)
                # Handle both 'date' and 'Date' column names
                date_col = 'date' if 'date' in self.price_data.columns else 'Date'
                self.price_data['Date'] = pd.to_datetime(self.price_data[date_col])
                self.price_data = self.price_data.set_index('Date')
                logger.info(f"Loaded {len(self.price_data)} price observations")

            # Load feature data (if available)
            feature_file = self.data_dir / f"{self.pair}_features.csv"
            if feature_file.exists():
                self.feature_data = pd.read_csv(feature_file)
                logger.info(f"Loaded {len(self.feature_data)} feature observations")

        except Exception as e:
            logger.error(f"Error loading data: {e}")

    def calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    model_type: str = 'regression') -> Dict:
        """
        Calculate comprehensive performance metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            model_type: 'regression' or 'classification'

        Returns:
            Dictionary of performance metrics
        """
        metrics = {}

        try:
            if model_type == 'regression':
                # Regression metrics
                metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
                metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                metrics['mape'] = float(mean_absolute_percentage_error(y_true, y_pred))
                metrics['r2'] = float(r2_score(y_true, y_pred))

                # Directional accuracy for financial returns
                if len(y_true) > 0 and len(y_pred) > 0:
                    direction_true = np.sign(y_true)
                    direction_pred = np.sign(y_pred)
                    metrics['directional_accuracy'] = float(accuracy_score(direction_true, direction_pred))

                # Profitability metrics
                returns = y_pred * y_true  # Simulated P&L
                metrics['total_return'] = float(np.sum(returns))
                metrics['win_rate'] = float(np.mean(returns > 0))
                metrics['profit_factor'] = float(
                    np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0]))
                ) if np.sum(returns < 0) != 0 else float('inf')

            elif model_type == 'classification':
                # Classification metrics
                y_pred_class = (y_pred > 0.5).astype(int) if y_pred.dtype == float else y_pred
                y_true_class = (y_true > 0).astype(int)

                metrics['accuracy'] = float(accuracy_score(y_true_class, y_pred_class))
                metrics['precision'] = float(precision_score(y_true_class, y_pred_class, zero_division=0))
                metrics['recall'] = float(recall_score(y_true_class, y_pred_class, zero_division=0))
                metrics['f1'] = float(f1_score(y_true_class, y_pred_class, zero_division=0))

                # Confusion matrix
                cm = confusion_matrix(y_true_class, y_pred_class)
                metrics['confusion_matrix'] = cm.tolist()

            # Additional metrics
            metrics['sample_size'] = len(y_true)

            # Statistical tests
            if len(y_true) > 10:
                # Normality test for residuals
                residuals = y_true - y_pred
                _, p_value = stats.shapiro(residuals)
                metrics['residuals_normal'] = p_value > 0.05

                # Mean error
                metrics['mean_error'] = float(np.mean(residuals))
                metrics['std_error'] = float(np.std(residuals))

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            metrics['error'] = str(e)

        return metrics

    def analyze_feature_importance(self, model_name: str = 'rf') -> Dict:
        """
        Analyze feature importance and contribution.

        Args:
            model_name: Name of the model to analyze

        Returns:
            Dictionary with feature importance analysis
        """
        analysis = {'feature_importance': {}, 'top_features': [], 'correlations': {}}

        try:
            if model_name not in self.models:
                return {'error': f'Model {model_name} not found'}

            model = self.models[model_name]

            # Get feature data
            if self.feature_data.empty:
                return {'error': 'No feature data available'}

            # Prepare data
            feature_cols = [col for col in self.feature_data.columns
                          if not col.startswith('target_') and col != 'Close']
            X = self.feature_data[feature_cols]
            y = self.feature_data.get('target_1d', self.feature_data.get('target_1d'))

            if y is None or y.empty:
                return {'error': 'No target variable found'}

            # Remove NaN values
            valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[valid_idx]
            y = y[valid_idx]

            if len(X) == 0:
                return {'error': 'No valid data after removing NaN'}

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = X.columns

                # Sort by importance
                indices = np.argsort(importances)[::-1]
                analysis['feature_importance'] = {
                    feature_names[i]: float(importances[i])
                    for i in indices[:20]  # Top 20 features
                }

                analysis['top_features'] = list(analysis['feature_importance'].keys())[:10]

            # Correlation analysis
            correlations = {}
            for col in X.columns[:20]:  # Limit to first 20 features
                corr = X[col].corr(y)
                if not np.isnan(corr):
                    correlations[col] = float(corr)

            analysis['correlations'] = dict(sorted(correlations.items(),
                                                 key=lambda x: abs(x[1]), reverse=True)[:10])

            # Permutation importance (if sklearn supports it)
            try:
                perm_importance = permutation_importance(model, X.values, y.values,
                                                        n_repeats=5, random_state=42)
                analysis['permutation_importance'] = {
                    X.columns[i]: float(perm_importance.importances_mean[i])
                    for i in range(len(X.columns))
                }
            except Exception as e:
                logger.warning(f"Could not calculate permutation importance: {e}")

        except Exception as e:
            logger.error(f"Error in feature importance analysis: {e}")
            analysis['error'] = str(e)

        return analysis

    def analyze_model_stability(self, model_name: str = 'rf', n_splits: int = 5) -> Dict:
        """
        Analyze model stability through cross-validation.

        Args:
            model_name: Name of the model to analyze
            n_splits: Number of cross-validation splits

        Returns:
            Dictionary with stability analysis
        """
        stability = {'cv_scores': [], 'stability_metrics': {}}

        try:
            if model_name not in self.models:
                return {'error': f'Model {model_name} not found'}

            model = self.models[model_name]

            # Get feature data
            if self.feature_data.empty:
                return {'error': 'No feature data available'}

            feature_cols = [col for col in self.feature_data.columns
                          if not col.startswith('target_') and col != 'Close']
            X = self.feature_data[feature_cols]
            y = self.feature_data.get('target_1d')

            if y is None or y.empty:
                return {'error': 'No target variable found'}

            # Remove NaN values
            valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[valid_idx]
            y = y[valid_idx]

            if len(X) < n_splits * 2:
                return {'error': 'Insufficient data for cross-validation'}

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)

            # Cross-validation scores
            cv_scores = cross_val_score(model, X.values, y.values, cv=tscv,
                                      scoring='neg_mean_absolute_error')
            stability['cv_scores'] = [-score for score in cv_scores]  # Convert to positive MAE

            # Stability metrics
            stability['stability_metrics'] = {
                'mean_cv_score': float(np.mean(stability['cv_scores'])),
                'std_cv_score': float(np.std(stability['cv_scores'])),
                'cv_score_variation': float(np.std(stability['cv_scores']) / np.mean(stability['cv_scores'])),
                'min_cv_score': float(np.min(stability['cv_scores'])),
                'max_cv_score': float(np.max(stability['cv_scores']))
            }

        except Exception as e:
            logger.error(f"Error in stability analysis: {e}")
            stability['error'] = str(e)

        return stability

    def analyze_prediction_errors(self, model_name: str = 'rf') -> Dict:
        """
        Analyze prediction errors and identify patterns.

        Args:
            model_name: Name of the model to analyze

        Returns:
            Dictionary with error analysis
        """
        error_analysis = {
            'error_distribution': {},
            'error_by_time': {},
            'error_by_magnitude': {},
            'systematic_biases': {}
        }

        try:
            if model_name not in self.models or 'predictions' not in self.models[model_name]:
                return {'error': f'No predictions available for {model_name}'}

            model_data = self.models[model_name]
            predictions = np.array(model_data['predictions'])
            actuals = np.array(model_data['actuals'])

            if len(predictions) != len(actuals):
                return {'error': 'Predictions and actuals length mismatch'}

            errors = actuals - predictions

            # Error distribution
            error_analysis['error_distribution'] = {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'min': float(np.min(errors)),
                'max': float(np.max(errors)),
                'skewness': float(stats.skew(errors)),
                'kurtosis': float(stats.kurtosis(errors))
            }

            # Error by time of day/week (if temporal data available)
            if hasattr(self, 'temporal_data') and self.temporal_data is not None:
                # This would analyze errors by time periods
                pass

            # Error by prediction magnitude
            pred_magnitudes = np.abs(predictions)
            magnitude_bins = pd.qcut(pred_magnitudes, q=5, duplicates='drop')
            error_by_magnitude = {}

            for bin_name, group in pd.DataFrame({'magnitude': magnitude_bins, 'error': errors}).groupby('magnitude'):
                error_by_magnitude[str(bin_name)] = {
                    'mean_error': float(np.mean(group['error'])),
                    'std_error': float(np.std(group['error'])),
                    'count': len(group)
                }

            error_analysis['error_by_magnitude'] = error_by_magnitude

            # Systematic biases
            large_errors = np.abs(errors) > np.std(errors) * 2
            if np.sum(large_errors) > 0:
                error_analysis['systematic_biases'] = {
                    'large_error_rate': float(np.mean(large_errors)),
                    'large_error_mean': float(np.mean(errors[large_errors])),
                    'large_error_direction': 'overprediction' if np.mean(errors[large_errors]) > 0 else 'underprediction'
                }

        except Exception as e:
            logger.error(f"Error in prediction error analysis: {e}")
            error_analysis['error'] = str(e)

        return error_analysis

    def analyze_calibration(self, model_name: str = 'rf') -> Dict:
        """
        Analyze prediction calibration and reliability.

        Args:
            model_name: Name of the model to analyze

        Returns:
            Dictionary with calibration analysis
        """
        calibration = {'calibration_curve': {}, 'reliability_metrics': {}}

        try:
            if 'calibrator' not in self.models:
                return {'error': 'No calibrator available'}

            calibrator = self.models['calibrator']

            # Get validation data
            if self.feature_data.empty:
                return {'error': 'No feature data available'}

            feature_cols = [col for col in self.feature_data.columns
                          if not col.startswith('target_') and col != 'Close']
            X = self.feature_data[feature_cols]
            y = self.feature_data.get('target_1d')

            if y is None or y.empty:
                return {'error': 'No target variable found'}

            # Remove NaN values
            valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[valid_idx]
            y = y[valid_idx]

            # Get calibrated predictions
            base_model = self.models.get(model_name, self.models.get('rf'))
            if base_model is None:
                return {'error': 'Base model not found'}

            base_predictions = base_model.predict_proba(X.values)[:, 1] if hasattr(base_model, 'predict_proba') else base_model.predict(X.values)
            calibrated_predictions = calibrator.predict_proba(base_predictions.reshape(-1, 1))[:, 1]

            # Calibration curve
            prob_true, prob_pred = calibration_curve((y > 0).astype(int), calibrated_predictions, n_bins=10)

            calibration['calibration_curve'] = {
                'predicted_probabilities': prob_pred.tolist(),
                'actual_frequencies': prob_true.tolist()
            }

            # Reliability metrics
            calibration['reliability_metrics'] = {
                'calibration_error': float(np.mean(np.abs(prob_pred - prob_true))),
                'brier_score': float(np.mean((calibrated_predictions - (y > 0).astype(int)) ** 2))
            }

        except Exception as e:
            logger.error(f"Error in calibration analysis: {e}")
            calibration['error'] = str(e)

        return calibration

    def generate_improvement_recommendations(self) -> List[Dict]:
        """
        Generate actionable recommendations for model improvement.

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        try:
            # Analyze current performance
            if 'performance' in self.diagnostic_report:
                perf = self.diagnostic_report['performance']

                # Performance-based recommendations
                if perf.get('directional_accuracy', 0) < 0.55:
                    recommendations.append({
                        'priority': 'high',
                        'category': 'performance',
                        'issue': 'Low directional accuracy',
                        'recommendation': 'Consider ensemble methods or additional features',
                        'expected_impact': 'high'
                    })

                if perf.get('mae', 0) > 0.01:
                    recommendations.append({
                        'priority': 'high',
                        'category': 'performance',
                        'issue': 'High prediction error',
                        'recommendation': 'Implement feature engineering or model regularization',
                        'expected_impact': 'high'
                    })

            # Feature importance analysis
            if 'feature_importance' in self.diagnostic_report:
                feat_imp = self.diagnostic_report['feature_importance']

                if 'top_features' in feat_imp and len(feat_imp['top_features']) < 5:
                    recommendations.append({
                        'priority': 'medium',
                        'category': 'features',
                        'issue': 'Limited feature importance',
                        'recommendation': 'Add more diverse technical indicators and fundamental data',
                        'expected_impact': 'medium'
                    })

            # Stability analysis
            if 'stability' in self.diagnostic_report:
                stab = self.diagnostic_report['stability']

                if stab.get('cv_score_variation', 0) > 0.5:
                    recommendations.append({
                        'priority': 'high',
                        'category': 'stability',
                        'issue': 'High model variance',
                        'recommendation': 'Implement regularization or ensemble methods',
                        'expected_impact': 'high'
                    })

            # Error analysis
            if 'error_analysis' in self.diagnostic_report:
                err = self.diagnostic_report['error_analysis']

                if not err.get('residuals_normal', True):
                    recommendations.append({
                        'priority': 'medium',
                        'category': 'errors',
                        'issue': 'Non-normal error distribution',
                        'recommendation': 'Consider robust loss functions or error transformation',
                        'expected_impact': 'medium'
                    })

            # Default recommendations
            if not recommendations:
                recommendations.extend([
                    {
                        'priority': 'medium',
                        'category': 'data',
                        'issue': 'Data quality check',
                        'recommendation': 'Implement automated data validation and cleaning',
                        'expected_impact': 'medium'
                    },
                    {
                        'priority': 'low',
                        'category': 'monitoring',
                        'issue': 'Model monitoring',
                        'recommendation': 'Set up automated model performance monitoring',
                        'expected_impact': 'low'
                    }
                ])

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append({
                'priority': 'high',
                'category': 'system',
                'issue': 'Diagnostic error',
                'recommendation': 'Review diagnostic framework implementation',
                'expected_impact': 'high'
            })

        return recommendations

    def generate_full_report(self) -> Dict:
        """
        Generate comprehensive diagnostic report.

        Returns:
            Dictionary containing full diagnostic analysis
        """
        logger.info(f"Generating full diagnostic report for {self.pair}")

        report = {
            'pair': self.pair,
            'timestamp': datetime.now().isoformat(),
            'models_analyzed': list(self.models.keys()),
            'data_summary': {
                'price_observations': len(self.price_data),
                'feature_observations': len(self.feature_data),
                'date_range': {
                    'start': self.price_data.index.min().isoformat() if not self.price_data.empty else None,
                    'end': self.price_data.index.max().isoformat() if not self.price_data.empty else None
                }
            }
        }

        # Performance analysis
        if 'rf' in self.models and not self.feature_data.empty:
            try:
                feature_cols = [col for col in self.feature_data.columns
                              if not col.startswith('target_') and col != 'Close']
                X = self.feature_data[feature_cols]
                y = self.feature_data.get('target_1d')

                if y is not None and not y.empty:
                    valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
                    X_valid = X[valid_idx]
                    y_valid = y[valid_idx]

                    if len(X_valid) > 0:
                        predictions = self.models['rf'].predict(X_valid.values)
                        report['performance'] = self.calculate_performance_metrics(
                            y_valid.values, predictions, 'regression'
                        )
            except Exception as e:
                logger.error(f"Error in performance analysis: {e}")
                report['performance_error'] = str(e)

        # Feature importance
        report['feature_importance'] = self.analyze_feature_importance('rf')

        # Model stability
        report['stability'] = self.analyze_model_stability('rf')

        # Error analysis
        report['error_analysis'] = self.analyze_prediction_errors('rf')

        # Calibration analysis
        if 'calibrator' in self.models:
            report['calibration'] = self.analyze_calibration('rf')

        # Improvement recommendations
        report['recommendations'] = self.generate_improvement_recommendations()

        # Summary statistics
        report['summary'] = {
            'overall_health': self._calculate_overall_health(report),
            'key_metrics': self._extract_key_metrics(report),
            'risk_assessment': self._assess_risks(report)
        }

        self.diagnostic_report = report

        # Save report
        self._save_report(report)

        return report

    def _calculate_overall_health(self, report: Dict) -> str:
        """Calculate overall model health score."""
        try:
            score = 0
            total_checks = 0

            # Performance checks
            if 'performance' in report:
                perf = report['performance']
                total_checks += 1
                if perf.get('directional_accuracy', 0) > 0.55:
                    score += 1

                total_checks += 1
                if perf.get('mae', 1) < 0.008:
                    score += 1

            # Stability checks
            if 'stability' in report:
                stab = report['stability']
                total_checks += 1
                if stab.get('cv_score_variation', 1) < 0.3:
                    score += 1

            # Feature checks
            if 'feature_importance' in report:
                feat = report['feature_importance']
                total_checks += 1
                if len(feat.get('top_features', [])) >= 5:
                    score += 1

            if total_checks == 0:
                return 'unknown'

            health_score = score / total_checks

            if health_score >= 0.8:
                return 'excellent'
            elif health_score >= 0.6:
                return 'good'
            elif health_score >= 0.4:
                return 'fair'
            else:
                return 'poor'

        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 'error'

    def _extract_key_metrics(self, report: Dict) -> Dict:
        """Extract key metrics for summary."""
        metrics = {}

        if 'performance' in report:
            perf = report['performance']
            metrics.update({
                'directional_accuracy': perf.get('directional_accuracy', 0),
                'mae': perf.get('mae', 0),
                'profit_factor': perf.get('profit_factor', 0)
            })

        if 'stability' in report:
            stab = report['stability']
            metrics['model_stability'] = 1 - stab.get('cv_score_variation', 1)

        return metrics

    def _assess_risks(self, report: Dict) -> Dict:
        """Assess model risks and limitations."""
        risks = {'level': 'low', 'issues': []}

        try:
            high_risk_issues = []
            medium_risk_issues = []

            # Performance risks
            if 'performance' in report:
                perf = report['performance']
                if perf.get('directional_accuracy', 0) < 0.5:
                    high_risk_issues.append('Poor directional accuracy')
                elif perf.get('directional_accuracy', 0) < 0.55:
                    medium_risk_issues.append('Below-average directional accuracy')

            # Stability risks
            if 'stability' in report:
                stab = report['stability']
                if stab.get('cv_score_variation', 0) > 0.5:
                    high_risk_issues.append('High model variance')
                elif stab.get('cv_score_variation', 0) > 0.3:
                    medium_risk_issues.append('Moderate model variance')

            # Data risks
            if report['data_summary']['price_observations'] < 500:
                medium_risk_issues.append('Limited historical data')

            # Set risk level
            if high_risk_issues:
                risks['level'] = 'high'
            elif medium_risk_issues:
                risks['level'] = 'medium'

            risks['issues'] = high_risk_issues + medium_risk_issues

        except Exception as e:
            logger.error(f"Error assessing risks: {e}")
            risks['level'] = 'unknown'
            risks['issues'] = ['Risk assessment error']

        return risks

    def _save_report(self, report: Dict):
        """Save diagnostic report to file."""
        try:
            report_file = self.output_dir / f"{self.pair}_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Diagnostic report saved to {report_file}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")

    def export_visualizations(self):
        """Export diagnostic visualizations."""
        try:
            if self.feature_data.empty:
                logger.warning("No feature data available for visualizations")
                return

            # Create visualizations directory
            viz_dir = self.output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)

            # Feature importance plot
            if 'feature_importance' in self.diagnostic_report:
                feat_imp = self.diagnostic_report['feature_importance']
                if 'feature_importance' in feat_imp:
                    plt.figure(figsize=(12, 8))
                    features = list(feat_imp['feature_importance'].keys())[:15]
                    importances = list(feat_imp['feature_importance'].values())[:15]

                    sns.barplot(x=importances, y=features)
                    plt.title(f'{self.pair} Feature Importance')
                    plt.xlabel('Importance')
                    plt.tight_layout()
                    plt.savefig(viz_dir / f"{self.pair}_feature_importance.png", dpi=300, bbox_inches='tight')
                    plt.close()

            # Error distribution plot
            if 'error_analysis' in self.diagnostic_report:
                err_analysis = self.diagnostic_report['error_analysis']
                if 'error_distribution' in err_analysis:
                    # This would create error distribution plots
                    pass

            logger.info(f"Visualizations exported to {viz_dir}")

        except Exception as e:
            logger.error(f"Error exporting visualizations: {e}")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Model Diagnostics Framework')
    parser.add_argument('--pair', required=True, help='Currency pair (EURUSD, XAUUSD)')
    parser.add_argument('--report', action='store_true', help='Generate full diagnostic report')
    parser.add_argument('--features', action='store_true', help='Analyze feature importance')
    parser.add_argument('--stability', action='store_true', help='Analyze model stability')
    parser.add_argument('--recommendations', action='store_true', help='Get improvement recommendations')
    parser.add_argument('--visualize', action='store_true', help='Export visualizations')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--models-dir', default='models', help='Models directory')
    parser.add_argument('--output-dir', default='diagnostics', help='Output directory')

    args = parser.parse_args()

    # Initialize diagnostics
    diagnostics = ModelDiagnosticsFramework(
        pair=args.pair,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        output_dir=args.output_dir
    )

    if args.report:
        # Generate full report
        report = diagnostics.generate_full_report()
        print(json.dumps(report, indent=2, default=str))

    elif args.features:
        # Feature importance analysis
        analysis = diagnostics.analyze_feature_importance()
        print(json.dumps(analysis, indent=2, default=str))

    elif args.stability:
        # Stability analysis
        analysis = diagnostics.analyze_model_stability()
        print(json.dumps(analysis, indent=2, default=str))

    elif args.recommendations:
        # Get recommendations
        recommendations = diagnostics.generate_improvement_recommendations()
        print(json.dumps(recommendations, indent=2, default=str))

    elif args.visualize:
        # Export visualizations
        diagnostics.export_visualizations()
        print("Visualizations exported")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()