#!/usr/bin/env python3
"""
Analyze Model Confidence Distribution

This script analyzes the confidence levels of model predictions
to find the optimal threshold for quality trading.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path.cwd()))

from scripts.forecasting import HybridPriceForecastingEnsemble
from lightgbm import LGBMClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_confidence_distribution(pair: str):
    """Analyze the confidence distribution and win rates at different thresholds"""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Analyzing {pair} Model Confidence Distribution")
    logger.info(f"{'='*80}")
    
    # Train model
    ensemble = HybridPriceForecastingEnsemble(pair)
    X_train, y_train, X_val, y_val = ensemble.load_and_prepare_datasets()
    
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    logger.info("Training model...")
    model.fit(X_train, y_train)
    
    # Get predictions and probabilities
    logger.info("Generating predictions...")
    probabilities = model.predict_proba(X_val)
    predictions = model.predict(X_val)
    
    # Calculate confidence (max probability)
    confidences = np.max(probabilities, axis=1)
    
    # Analyze distribution
    logger.info(f"\n📊 CONFIDENCE DISTRIBUTION:")
    logger.info(f"   Mean: {confidences.mean():.3f}")
    logger.info(f"   Median: {np.median(confidences):.3f}")
    logger.info(f"   Std: {confidences.std():.3f}")
    logger.info(f"   Min: {confidences.min():.3f}")
    logger.info(f"   Max: {confidences.max():.3f}")
    
    # Percentiles
    percentiles = [50, 60, 70, 75, 80, 85, 90, 95]
    logger.info(f"\n📈 CONFIDENCE PERCENTILES:")
    for p in percentiles:
        value = np.percentile(confidences, p)
        count = (confidences >= value).sum()
        pct = count / len(confidences) * 100
        logger.info(f"   {p}th percentile: {value:.3f} ({count} predictions, {pct:.1f}%)")
    
    # Win rate at different thresholds
    logger.info(f"\n🎯 WIN RATE AT DIFFERENT CONFIDENCE THRESHOLDS:")
    
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80]
    best_threshold = None
    best_win_rate = 0
    
    results = []
    
    for threshold in thresholds:
        mask = confidences >= threshold
        if mask.sum() == 0:
            logger.info(f"   Confidence >= {threshold:.0%}: No predictions")
            continue
        
        # Calculate win rate for these high-confidence predictions
        high_conf_predictions = predictions[mask]
        high_conf_actuals = y_val.values[mask]
        
        win_rate = (high_conf_predictions == high_conf_actuals).mean()
        count = mask.sum()
        trades_per_month = count / (len(X_val) / 252) / 12
        
        results.append({
            'threshold': threshold,
            'win_rate': win_rate,
            'count': count,
            'trades_per_month': trades_per_month
        })
        
        logger.info(f"   Confidence >= {threshold:.0%}: {win_rate:.1%} win rate ({count} predictions, ~{trades_per_month:.1f} trades/month)")
        
        # Track best threshold with reasonable trade frequency
        if win_rate > best_win_rate and trades_per_month >= 2:
            best_win_rate = win_rate
            best_threshold = threshold
    
    # Recommendation
    logger.info(f"\n💡 RECOMMENDATION:")
    if best_threshold:
        logger.info(f"   Best Threshold: {best_threshold:.0%}")
        logger.info(f"   Expected Win Rate: {best_win_rate:.1%}")
        best_result = [r for r in results if r['threshold'] == best_threshold][0]
        logger.info(f"   Expected Trades/Month: ~{best_result['trades_per_month']:.1f}")
    else:
        # Find threshold with acceptable win rate (>60%)
        acceptable = [r for r in results if r['win_rate'] >= 0.60 and r['trades_per_month'] >= 1]
        if acceptable:
            best = max(acceptable, key=lambda x: x['win_rate'])
            logger.info(f"   Suggested Threshold: {best['threshold']:.0%}")
            logger.info(f"   Expected Win Rate: {best['win_rate']:.1%}")
            logger.info(f"   Expected Trades/Month: ~{best['trades_per_month']:.1f}")
        else:
            logger.info(f"   Model may need improvement - no threshold achieves >60% win rate with reasonable frequency")
    
    return results


def main():
    """Analyze both pairs"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║        MODEL CONFIDENCE DISTRIBUTION ANALYSIS                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    for pair in ['EURUSD', 'XAUUSD']:
        try:
            results = analyze_confidence_distribution(pair)
        except Exception as e:
            logger.error(f"Error analyzing {pair}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
