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
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, '/app')

from scripts.optimizer import optimize_pair
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
        """Optimize model until target accuracy is reached"""
        logger.info(f"Starting automated optimization for {pair} targeting {self.target_accuracy}")

        best_accuracy = 0
        iteration = 0
        results_history = []

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{self.max_iterations} for {pair}")

            try:
                # Run optimization
                improvement = optimize_pair(pair, threshold=self.target_accuracy - 0.1)
                logger.info(f"Optimization completed for {pair}, improvement: {improvement}")

                # Evaluate performance
                performance = self.evaluate_current_performance(pair)
                current_accuracy = performance.get('accuracy', 0)

                results_history.append({
                    'iteration': iteration,
                    'accuracy': current_accuracy,
                    'improvement': improvement,
                    'timestamp': datetime.now().isoformat()
                })

                logger.info(f"{pair} accuracy after iteration {iteration}: {current_accuracy:.4f}")

                # Check if target reached
                if current_accuracy >= self.target_accuracy:
                    logger.info(f"Target accuracy {self.target_accuracy} reached for {pair}!")
                    self.notifier.send_notification(
                        subject=f"🎯 Target Accuracy Reached for {pair}",
                        message=f"Model optimization completed!\n\n"
                               f"Pair: {pair}\n"
                               f"Final Accuracy: {current_accuracy:.4f}\n"
                               f"Target: {self.target_accuracy}\n"
                               f"Iterations: {iteration}\n"
                               f"Profit Factor: {performance.get('profit_factor', 'N/A')}\n"
                               f"Total Trades: {performance.get('total_trades', 'N/A')}"
                    )
                    break

                # Save progress
                self.save_progress(pair, results_history)

                # Small delay between iterations
                time.sleep(5)

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