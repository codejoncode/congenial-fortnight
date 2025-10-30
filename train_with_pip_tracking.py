#!/usr/bin/env python3
"""
Complete Training with Pip-Based Signal Evaluation

This script:
1. Trains models using existing pipeline
2. Generates predictions on validation/test data
3. Evaluates using pip-based quality system
4. Reports detailed pip statistics

Focus: Quality setups with 75%+ win rate, 2:1+ R:R
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging
from datetime import datetime
import joblib

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.forecasting import HybridPriceForecastingEnsemble
from scripts.pip_based_signal_system import PipBasedSignalSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipBasedTraining:
    """
    Enhanced training system with pip-based evaluation
    """
    
    def __init__(self, pairs: list = ['EURUSD', 'XAUUSD'], confidence_threshold: float = 0.70):
        self.pairs = pairs
        self.models = {}
        self.confidence_threshold = confidence_threshold
        self.pip_system = PipBasedSignalSystem(
            min_risk_reward=2.0,  # Minimum 1:2 R:R
            min_confidence=confidence_threshold
        )
        
    def train_and_evaluate(self):
        """
        Complete training and pip-based evaluation pipeline
        """
        
        print("\n" + "="*80)
        print("🎯 TRAINING WITH PIP-BASED QUALITY EVALUATION")
        print("="*80)
        print(f"\nPairs: {', '.join(self.pairs)}")
        print(f"Min Risk:Reward: 1:{self.pip_system.min_risk_reward}")
        print(f"Min Confidence: {self.pip_system.min_confidence*100:.0f}%")
        print(f"Trading Strategy: Quality setups only (not every day)")
        print(f"\n📊 Expected Performance (based on confidence analysis):")
        if self.confidence_threshold == 0.70:
            print(f"   EURUSD: ~76% win rate, ~10 trades/month")
            print(f"   XAUUSD: ~85% win rate, ~15 trades/month")
        elif self.confidence_threshold == 0.75:
            print(f"   EURUSD: ~83% win rate, ~8.5 trades/month")
            print(f"   XAUUSD: ~88% win rate, ~14 trades/month")
        elif self.confidence_threshold == 0.80:
            print(f"   EURUSD: ~89% win rate, ~7 trades/month")
            print(f"   XAUUSD: ~92% win rate, ~13 trades/month")
        print("="*80 + "\n")
        
        for pair in self.pairs:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing {pair}")
            logger.info(f"{'='*80}")
            
            try:
                # Step 1: Train model
                logger.info(f"📚 Step 1/5: Training {pair} model...")
                model, train_acc, val_acc, X_train, X_val, y_train, y_val = self._train_model(pair)
                self.models[pair] = model
                
                logger.info(f"   Training Accuracy: {train_acc*100:.2f}%")
                logger.info(f"   Validation Accuracy: {val_acc*100:.2f}%")
                
                # Step 2: Load full data for backtesting
                logger.info(f"\n📊 Step 2/5: Loading historical data...")
                ensemble = HybridPriceForecastingEnsemble(pair)
                
                # Get the price data (daily OHLC)
                # The ensemble already has loaded this in __init__
                if hasattr(ensemble, 'daily_data') and not ensemble.daily_data.empty:
                    full_data = ensemble.daily_data
                elif hasattr(ensemble, 'price_data') and not ensemble.price_data.empty:
                    full_data = ensemble.price_data
                else:
                    # Fallback: load daily file directly
                    daily_file = Path('data') / f'{pair}_Daily.csv'
                    full_data = pd.read_csv(daily_file)
                    full_data['time'] = pd.to_datetime(full_data['timestamp'])
                    full_data.set_index('time', inplace=True)
                
                # Use validation period for pip-based backtest
                val_size = len(X_val)
                val_data = full_data.iloc[-val_size:].copy()
                
                # Ensure we have OHLC columns with proper capitalization
                if 'close' in val_data.columns:
                    val_data.rename(columns={
                        'open': 'Open', 'high': 'High', 
                        'low': 'Low', 'close': 'Close'
                    }, inplace=True)
                
                logger.info(f"   Validation period: {val_data.index[0]} to {val_data.index[-1]}")
                logger.info(f"   Total candles: {len(val_data)}")
                
                # Step 3: Generate predictions with confidence scores
                logger.info(f"\n🔮 Step 3/5: Generating quality signals...")
                signals = self._generate_quality_signals(pair, model, val_data, X_val)
                
                quality_signals = [s for s in signals if s['signal'] is not None]
                logger.info(f"   Total predictions: {len(signals)}")
                logger.info(f"   Quality setups detected: {len(quality_signals)}")
                logger.info(f"   Signal frequency: {len(quality_signals)/len(signals)*100:.1f}%")
                
                # Step 4: Backtest with pip tracking
                logger.info(f"\n💰 Step 4/5: Backtesting with pip tracking...")
                pip_results = self.pip_system.backtest_with_pip_tracking(
                    val_data, pair, quality_signals
                )
                
                # Step 5: Display results
                logger.info(f"\n📈 Step 5/5: Results summary")
                self.pip_system.print_backtest_summary(pip_results)
                
                # Check if meets quality criteria
                self._evaluate_system_quality(pip_results)
                
            except Exception as e:
                logger.error(f"❌ Error processing {pair}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save all results
        logger.info(f"\n💾 Saving detailed results...")
        self.pip_system.save_detailed_results('output/pip_results')
        
        # Final summary
        self._print_final_summary()
    
    def _train_model(self, pair: str):
        """Train model using existing pipeline"""
        
        ensemble = HybridPriceForecastingEnsemble(pair)
        X_train, y_train, X_val, y_val = ensemble.load_and_prepare_datasets()
        
        # Train LightGBM model
        from lightgbm import LGBMClassifier
        
        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        logger.info(f"   Training samples: {len(X_train)}")
        logger.info(f"   Validation samples: {len(X_val)}")
        logger.info(f"   Features: {X_train.shape[1]}")
        
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)
        
        # Save model
        model_path = Path('models') / f'{pair}_pip_based_model.joblib'
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"   ✅ Model saved to {model_path}")
        
        return model, train_acc, val_acc, X_train, X_val, y_train, y_val
    
    def _generate_quality_signals(self, pair: str, model, val_data: pd.DataFrame,
                                  X_val: pd.DataFrame) -> list:
        """
        Generate signals using pip-based quality system
        """
        signals = []
        
        # Get model predictions with probabilities
        predictions = model.predict(X_val)
        probabilities = model.predict_proba(X_val)
        
        logger.info(f"   Generating quality signals from {len(predictions)} predictions...")
        
        # We need to use the full historical data for each point
        # val_data is the price data, but we need to align it with X_val indices
        
        # For simplicity, let's just use a sliding window approach
        # We'll check quality for each prediction point
        quality_count = 0
        
        for i in range(len(predictions)):
            try:
                # Model prediction
                pred = predictions[i]
                prob = probabilities[i]
                
                model_prediction = {
                    'direction': 'long' if pred == 1 else 'short',
                    'confidence': max(prob)
                }
                
                # For quality check, we need at least 50 bars of history
                if i < 50:
                    continue
                
                # Get historical window (last 200 bars for context)
                start_idx = max(0, i - 200)
                historical_data = val_data.iloc[start_idx:i+1].copy()
                
                if len(historical_data) < 50:
                    continue
                
                # Check if this is a quality setup
                signal = self.pip_system.detect_quality_setup(
                    historical_data,
                    pair,
                    model_prediction
                )
                
                # Add timestamp from validation data
                if signal and signal['signal'] is not None:
                    signal['timestamp'] = val_data.index[i] if hasattr(val_data.index, '__getitem__') else i
                    quality_count += 1
                
                signals.append(signal)
                
            except Exception as e:
                logger.debug(f"Error generating signal at index {i}: {e}")
                continue
        
        logger.info(f"   Generated {quality_count} quality setups from {len(signals)} predictions")
        
        return signals
    
    def _evaluate_system_quality(self, results: dict):
        """
        Evaluate if system meets quality criteria
        """
        print("\n" + "="*80)
        print("🎯 QUALITY CRITERIA EVALUATION")
        print("="*80)
        
        # Criteria
        target_win_rate = 0.75
        target_risk_reward = 2.0
        target_monthly_trades = 3  # At least 3 quality setups per month
        
        win_rate = results['win_rate']
        avg_rr = results['avg_risk_reward']
        monthly_trades = results['trades_per_month']
        
        # Check each criterion
        win_rate_pass = win_rate >= target_win_rate
        rr_pass = avg_rr >= target_risk_reward
        frequency_pass = monthly_trades >= target_monthly_trades
        
        print(f"\n✅ Target: {target_win_rate*100:.0f}%+ win rate")
        print(f"   Actual: {win_rate*100:.1f}% {'✅ PASS' if win_rate_pass else '❌ FAIL'}")
        
        print(f"\n✅ Target: {target_risk_reward}:1+ Risk:Reward")
        print(f"   Actual: 1:{avg_rr:.2f} {'✅ PASS' if rr_pass else '❌ FAIL'}")
        
        print(f"\n✅ Target: {target_monthly_trades}+ trades/month")
        print(f"   Actual: {monthly_trades:.1f} {'✅ PASS' if frequency_pass else '❌ FAIL'}")
        
        # Overall evaluation
        all_pass = win_rate_pass and rr_pass and frequency_pass
        
        if all_pass:
            print(f"\n🎉 SYSTEM PASSES ALL QUALITY CRITERIA!")
        else:
            print(f"\n⚠️  System needs improvement in flagged areas")
        
        # Calculate expected profit
        if results['total_trades'] > 0:
            expectancy = (
                results['win_rate'] * results['avg_win_pips'] +
                (1 - results['win_rate']) * results['avg_loss_pips']
            )
            
            expected_monthly_pips = expectancy * monthly_trades
            
            print(f"\n💰 EXPECTED PERFORMANCE:")
            print(f"   Expectancy: {expectancy:+.2f} pips per trade")
            print(f"   Expected Monthly Pips: {expected_monthly_pips:+.1f}")
            
            # Note: Dollar calculation would vary by pair and lot size
        
        print("="*80 + "\n")
    
    def _print_final_summary(self):
        """Print final summary across all pairs"""
        
        print("\n" + "="*80)
        print("📊 FINAL SUMMARY - ALL PAIRS")
        print("="*80)
        
        all_results = self.pip_system.backtest_results
        
        if not all_results:
            print("No results to display")
            return
        
        # Create summary table
        summary_data = []
        for r in all_results:
            summary_data.append({
                'Pair': r['pair'],
                'Trades': r['total_trades'],
                'Win Rate': f"{r['win_rate']*100:.1f}%",
                'Total Pips': f"{r['total_pips']:+.1f}",
                'Avg Win': f"{r['avg_win_pips']:+.1f}",
                'Avg Loss': f"{r['avg_loss_pips']:+.1f}",
                'Avg R:R': f"1:{r['avg_risk_reward']:.2f}",
                'Trades/Month': f"{r['trades_per_month']:.1f}"
            })
        
        df_summary = pd.DataFrame(summary_data)
        print("\n" + df_summary.to_string(index=False))
        
        # Overall stats
        total_trades = sum(r['total_trades'] for r in all_results)
        total_pips = sum(r['total_pips'] for r in all_results)
        
        print(f"\n{'='*80}")
        print(f"OVERALL:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Total Pips: {total_pips:+.1f}")
        print(f"  Average Pips Per Trade: {total_pips/total_trades if total_trades > 0 else 0:+.2f}")
        print(f"{'='*80}\n")


def main():
    """
    Run complete training and pip-based evaluation
    """
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     FOREX TRAINING WITH PIP-BASED QUALITY EVALUATION         ║
    ╚══════════════════════════════════════════════════════════════╝
    
    This system trains models and evaluates using pip-based metrics:
    
    ✅ Risk:Reward: Minimum 1:2 ratio
    ✅ Win Rate: Target 75%+
    ✅ Frequency: Quality setups only (not every day)
    ✅ Comprehensive pip tracking
    
    Results include:
    - Total pips won/lost over backtest period
    - Average pips per winning trade
    - Average pips per losing trade
    - Win rate percentage
    - Risk:Reward ratios
    - Trades per month frequency
    - Detailed trade-by-trade breakdown
    
    Confidence Threshold Options:
    - 70%: Higher frequency (~10-15 trades/month), 76-85% win rate
    - 75%: Balanced (~8-14 trades/month), 83-88% win rate  
    - 80%: Ultra-selective (~7-13 trades/month), 89-92% win rate
    
    """)
    
    # Configure pairs and confidence threshold
    import sys
    confidence = 0.70  # Default: balanced approach
    
    if len(sys.argv) > 1:
        try:
            confidence = float(sys.argv[1])
            if confidence < 0.5 or confidence > 0.95:
                print("⚠️  Confidence must be between 0.50 and 0.95")
                confidence = 0.70
        except:
            print("⚠️  Invalid confidence value, using default 0.70")
    
    pairs = ['EURUSD', 'XAUUSD']
    
    print(f"🎯 Using confidence threshold: {confidence*100:.0f}%\n")
    
    # Create trainer
    trainer = PipBasedTraining(pairs=pairs, confidence_threshold=confidence)
    
    # Run complete pipeline
    trainer.train_and_evaluate()
    
    print("\n✅ Training and evaluation complete!")
    print("📁 Detailed results saved in output/pip_results/")
    print(f"\n💡 To try different confidence threshold:")
    print(f"   python train_with_pip_tracking.py 0.75  (for 75%)")
    print(f"   python train_with_pip_tracking.py 0.80  (for 80%)")


if __name__ == '__main__':
    main()
