from django.core.management.base import BaseCommand
import sys
import os
sys.path.append(os.getcwd())
from daily_forex_signal_system import DailyForexSignal

class Command(BaseCommand):
    help = 'Run backtest to analyze probability vs win/loss rates'

    def add_arguments(self, parser):
        parser.add_argument('pair', type=str, help='Pair to backtest (EURUSD or XAUUSD)')
        parser.add_argument('--days', type=int, default=60, help='Number of days to backtest')

    def handle(self, *args, **options):
        pair = options['pair']
        days = options['days']
        
        ds = DailyForexSignal()
        result = ds.backtest_last_n_days(pair, n=days)
        
        self.stdout.write(f"Backtest Results for {pair} ({days} days):")
        self.stdout.write(f"Total Signals: {result['total_signals']}")
        self.stdout.write(f"Overall Accuracy: {result['accuracy']:.1%}")
        self.stdout.write(f"Bullish Signals: {result['bullish']} ({result['bullish_accuracy']:.1f}% accuracy)")
        self.stdout.write(f"Bearish Signals: {result['bearish']} ({result['bearish_accuracy']:.1f}% accuracy)")
        self.stdout.write(f"No Signals: {result['no_signal']}")
        
        # To analyze by probability, we'd need to modify the backtest to track probabilities
        self.stdout.write("\nTo analyze by probability ranges, modify backtest_last_n_days to return probability data.")