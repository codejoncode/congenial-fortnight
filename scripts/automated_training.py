#!/usr/bin/env python3
"""
Automated Training Script for Cloud Run Jobs
Continuously optimizes models until target accuracy is reached
"""

import os
import sys
import json
import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, '/app')

try:
    from scripts.advanced_regularization_optimizer import optimize_pair
except ImportError:
    from scripts.optimizer import optimize_pair

try:
    from scripts.regularization_config_manager import get_regularization_config
except ImportError:
    get_regularization_config = None

from scripts.forecasting import ForecastingSystem
from notification_system import NotificationSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/automated_training.log')
    ]
)
logger = logging.getLogger(__name__)

class AutomatedTrainer:
    def __init__(self, target_accuracy: float = 0.85, max_iterations: int = 50):
        self.target_accuracy = target_accuracy
        self.max_iterations = max_iterations
        self.forecasting = ForecastingSystem()
        self.notifier = NotificationSystem()

        # Enhanced stopping criteria
        self.convergence_patience = 10  # Iterations without improvement
        self.min_improvement = 0.001    # Minimum improvement threshold
        self.early_stop_threshold = 0.95  # Stop if we exceed target significantly
        
        # Performance tracking
        self.performance_history = {}
        self.stagnation_counters = {}

        # Ensure directories exist
        os.makedirs('/app/models', exist_ok=True)
        os.makedirs('/app/logs', exist_ok=True)
        os.makedirs('/app/output', exist_ok=True)

    def evaluate_current_performance(self, pair: str) -> Dict:
        """Evaluate current model performance"""
        try:
            # Load model and run backtest
            results = self.forecasting.backtest_ensemble(
                pair=pair,
                days=30,  # Use last 30 days for evaluation
                save_results=False
            )
            return {
                'accuracy': results.get('directional_accuracy', 0),
                'profit_factor': results.get('profit_factor', 0),
                'total_trades': results.get('total_trades', 0),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error evaluating {pair}: {e}")
            return {'accuracy': 0, 'error': str(e)}

    def optimize_until_target(self, pair: str) -> Dict:
        """Optimize model until target accuracy is reached with advanced early stopping"""
        logger.info(f"Starting automated optimization for {pair} targeting {self.target_accuracy}")

        best_accuracy = 0
        iteration = 0
        results_history = []
        stagnation_counter = 0
        convergence_window = []
        
        # Initialize tracking for this pair
        self.performance_history[pair] = []
        self.stagnation_counters[pair] = 0

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{self.max_iterations} for {pair}")

            try:
                # Get adaptive regularization configuration
                if get_regularization_config is not None:
                    reg_config = get_regularization_config(
                        pair, 
                        target_accuracy=self.target_accuracy,
                        current_performance=current_accuracy if 'current_accuracy' in locals() else None,
                        iteration=iteration
                    )
                    logger.info(f"Using {reg_config.get('meta', {}).get('strategy', 'default')} regularization strategy for {pair}")
                
                # Run optimization with enhanced parameters
                improvement = optimize_pair(pair, threshold=self.target_accuracy - 0.1)
                logger.info(f"Optimization completed for {pair}, improvement: {improvement}")

                # Evaluate performance
                performance = self.evaluate_current_performance(pair)
                current_accuracy = performance.get('accuracy', 0)

                # Track performance history
                self.performance_history[pair].append(current_accuracy)
                
                # Calculate improvement from best
                improvement_from_best = current_accuracy - best_accuracy
                
                results_history.append({
                    'iteration': iteration,
                    'accuracy': current_accuracy,
                    'improvement': improvement,
                    'improvement_from_best': improvement_from_best,
                    'timestamp': datetime.now().isoformat()
                })

                logger.info(f"{pair} accuracy after iteration {iteration}: {current_accuracy:.4f} (improvement: {improvement_from_best:+.4f})")

                # Advanced Early Stopping Logic
                
                # 1. Check for exceptional performance (stop early if significantly exceeding target)
                if current_accuracy >= self.early_stop_threshold:
                    logger.info(f"🚀 Exceptional performance reached for {pair}: {current_accuracy:.4f} >= {self.early_stop_threshold:.4f}")
                    break
                
                # 2. Check for convergence (no significant improvement)
                if improvement_from_best < self.min_improvement:
                    stagnation_counter += 1
                    logger.info(f"Stagnation counter for {pair}: {stagnation_counter}/{self.convergence_patience}")
                else:
                    stagnation_counter = 0
                    best_accuracy = max(best_accuracy, current_accuracy)
                
                # 3. Early stopping due to convergence
                if stagnation_counter >= self.convergence_patience:
                    logger.info(f"⏹️ Early stopping triggered for {pair} due to convergence (no improvement for {self.convergence_patience} iterations)")
                    break
                
                # 4. Check convergence window (variance in recent performance)
                convergence_window.append(current_accuracy)
                if len(convergence_window) > 5:  # Keep last 5 results
                    convergence_window.pop(0)
                    
                if len(convergence_window) >= 5:
                    recent_variance = np.var(convergence_window)
                    if recent_variance < 0.0001:  # Very low variance indicates convergence
                        logger.info(f"🎯 Performance convergence detected for {pair} (variance: {recent_variance:.6f})")
                        # Don't break immediately, but increase stagnation counter
                        stagnation_counter += 2

                # Send progress notifications
                if iteration % 3 == 0 or (improvement_from_best > 0.01):  # More frequent updates
                    self.send_progress_notification(pair, iteration, current_accuracy, best_accuracy)

                # Check if target reached
                if current_accuracy >= self.target_accuracy:
                    logger.info(f"Target accuracy {self.target_accuracy} reached for {pair}!")
                    self.notifier.send_notification(
                        subject=f"🎯 TARGET ACCURACY REACHED: {pair} - {current_accuracy:.1%}",
                        message=f"🚀 SUCCESS! Model optimization completed!\n\n"
                               f"🎯 Pair: {pair}\n"
                               f"📊 Final Accuracy: {current_accuracy:.1%}\n"
                               f"🎯 Target: {self.target_accuracy:.1%}\n"
                               f"🔄 Iterations: {iteration}\n"
                               f"💰 Profit Factor: {performance.get('profit_factor', 'N/A')}\n"
                               f"📈 Total Trades: {performance.get('total_trades', 'N/A')}\n\n"
                               f"✅ Automated training will continue for other pairs.",
                        email_recipient=os.getenv('NOTIFICATION_EMAIL', 'mydecretor@protonmail.com')
                    )
                    break

                # Advanced Performance Analysis
                if len(results_history) >= 3:
                    recent_performance = [r['accuracy'] for r in results_history[-3:]]
                    performance_trend = self._analyze_performance_trend(recent_performance)
                    
                    logger.info(f"Performance trend for {pair}: {performance_trend}")
                    
                    # Adaptive strategy adjustment
                    if performance_trend == 'declining' and iteration > 10:
                        logger.warning(f"Declining performance detected for {pair}. Consider strategy adjustment.")
                    elif performance_trend == 'plateauing' and stagnation_counter > 5:
                        logger.info(f"Performance plateau detected for {pair}. Increasing regularization focus.")

                # Save progress with enhanced metrics
                self.save_progress(pair, results_history)

                # Dynamic delay based on performance
                if improvement_from_best > 0.01:
                    time.sleep(2)  # Shorter delay when making good progress
                else:
                    time.sleep(5)  # Standard delay

            except Exception as e:
                logger.error(f"Error in iteration {iteration} for {pair}: {e}")
                continue

        return {
            'pair': pair,
            'final_accuracy': current_accuracy,
            'iterations_completed': iteration,
            'target_reached': current_accuracy >= self.target_accuracy,
            'results_history': results_history
        }

    def send_progress_notification(self, pair: str, iteration: int, current_accuracy: float, best_accuracy: float):
        """Send progress update notification during training"""
        progress_message = f"🤖 Training Progress Update\n\n"
        progress_message += f"📊 Pair: {pair}\n"
        progress_message += f"🔄 Iteration: {iteration}/{self.max_iterations}\n"
        progress_message += f"📈 Current Accuracy: {current_accuracy:.1%}\n"
        progress_message += f"🎯 Best Accuracy: {best_accuracy:.1%}\n"
        progress_message += f"🎯 Target: {self.target_accuracy:.1%}\n"
        progress_message += f"⏱️  Time: {datetime.now().strftime('%H:%M:%S')}\n\n"
        
        if current_accuracy >= self.target_accuracy:
            progress_message += "🎉 TARGET ACHIEVED!\n"
        else:
            progress_remaining = self.target_accuracy - current_accuracy
            progress_message += f"📍 Remaining: {progress_remaining:.1%}\n"
            progress_message += "🔄 Training continues..."
        
        self.notifier.send_notification(
            subject=f"🤖 {pair} Training Progress - {current_accuracy:.1%}",
            message=progress_message,
            email_recipient=os.getenv('NOTIFICATION_EMAIL', 'mydecretor@protonmail.com')
        )

    def save_progress(self, pair: str, results_history: List[Dict]):
        """Save optimization progress"""
        progress_file = f'/app/logs/optimization_progress_{pair}.json'
        try:
            with open(progress_file, 'w') as f:
                json.dump({
                    'pair': pair,
                    'target_accuracy': self.target_accuracy,
                    'last_updated': datetime.now().isoformat(),
                    'results_history': results_history
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving progress for {pair}: {e}")

    def _analyze_performance_trend(self, recent_performance: List[float]) -> str:
        """Analyze recent performance trend."""
        if len(recent_performance) < 3:
            return 'insufficient_data'
        
        # Calculate trend
        improvements = [recent_performance[i] - recent_performance[i-1] for i in range(1, len(recent_performance))]
        
        avg_improvement = np.mean(improvements)
        
        if avg_improvement > 0.005:
            return 'improving'
        elif avg_improvement < -0.005:
            return 'declining'
        else:
            return 'plateauing'

    def run_automated_training(self, pairs: List[str] = None):
        """Run automated training for specified pairs"""
        if pairs is None:
            pairs = ['EURUSD', 'XAUUSD']

        logger.info(f"Starting automated training for pairs: {pairs}")
        logger.info(f"Target accuracy: {self.target_accuracy}")

        results = {}
        start_time = datetime.now()

        for pair in pairs:
            try:
                result = self.optimize_until_target(pair)
                results[pair] = result

                # Send progress notification
                self.notifier.send_notification(
                    subject=f"📊 {pair} Optimization Progress",
                    message=f"Completed optimization for {pair}\n\n"
                           f"Final Accuracy: {result['final_accuracy']:.4f}\n"
                           f"Target: {self.target_accuracy}\n"
                           f"Iterations: {result['iterations_completed']}\n"
                           f"Target Reached: {'✅' if result['target_reached'] else '❌'}"
                )

            except Exception as e:
                logger.error(f"Failed to optimize {pair}: {e}")
                results[pair] = {'error': str(e)}

        end_time = datetime.now()
        duration = end_time - start_time

        # Send final summary
        summary_message = f"🤖 Automated Training Complete\n\n"
        summary_message += f"Duration: {duration}\n"
        summary_message += f"Target Accuracy: {self.target_accuracy}\n\n"

        for pair, result in results.items():
            if 'error' in result:
                summary_message += f"❌ {pair}: ERROR - {result['error']}\n"
            else:
                status = "✅" if result.get('target_reached', False) else "❌"
                summary_message += f"{status} {pair}: {result.get('final_accuracy', 0):.4f} ({result.get('iterations_completed', 0)} iterations)\n"

        self.notifier.send_notification(
            subject="🎯 Automated Training Summary",
            message=summary_message
        )

        # Save final results
        final_results = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'target_accuracy': self.target_accuracy,
            'results': results
        }

        with open('/app/logs/automated_training_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)

        logger.info("Automated training completed")
        return final_results

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Automated Model Training')
    parser.add_argument('--target', type=float, default=0.85,
                       help='Target accuracy (default: 0.85)')
    parser.add_argument('--max-iterations', type=int, default=50,
                       help='Maximum iterations per pair (default: 50)')
    parser.add_argument('--pairs', nargs='+', default=['EURUSD', 'XAUUSD'],
                       help='Currency pairs to optimize (default: EURUSD XAUUSD)')

    args = parser.parse_args()

    trainer = AutomatedTrainer(
        target_accuracy=args.target,
        max_iterations=args.max_iterations
    )

    results = trainer.run_automated_training(args.pairs)

    # Exit with success/failure code
    all_targets_reached = all(
        result.get('target_reached', False)
        for result in results.get('results', {}).values()
        if 'error' not in result
    )

    sys.exit(0 if all_targets_reached else 1)

if __name__ == '__main__':
    main()