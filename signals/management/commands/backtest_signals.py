from django.core.management.base import BaseCommand
import sys
import os
sys.path.append(os.getcwd())
from daily_forex_signal_system import DailyForexSignal

class Command(BaseCommand):
    help = 'Run enhanced backtest to analyze probability vs win/loss rates with pips analysis'

    def add_arguments(self, parser):
        parser.add_argument('pair', type=str, help='Pair to backtest (EURUSD or XAUUSD)')
        parser.add_argument('--days', type=int, default=60, help='Number of days to backtest')
        parser.add_argument('--export-csv', action='store_true', help='Export detailed results to CSV file')

    def handle(self, *args, **options):
        pair = options['pair']
        days = options['days']
        export_csv = options['export_csv']

        ds = DailyForexSignal()
        result = ds.backtest_last_n_days_enhanced(pair, n=days)

        self.stdout.write(f"Enhanced Backtest Results for {pair} ({days} days):")
        self.stdout.write(f"Total Signals: {result['total_signals']}")
        self.stdout.write(f"Overall Accuracy: {result['accuracy']:.1f}%")
        self.stdout.write(f"Bullish Signals: {result['bullish']} ({result['bullish_accuracy']:.1f}% accuracy)")
        self.stdout.write(f"Bearish Signals: {result['bearish']} ({result['bearish_accuracy']:.1f}% accuracy)")
        self.stdout.write(f"No Signals: {result['no_signal']}")

        # Enhanced pips analysis
        self.stdout.write(f"\n=== PIPS ANALYSIS ===")
        self.stdout.write(f"Total Pips Won: {result['total_pips_won']:.1f}")
        self.stdout.write(f"Total Pips Lost: {result['total_pips_lost']:.1f}")
        self.stdout.write(f"Net Pips: {result['net_pips']:.1f}")
        self.stdout.write(f"Average Win: {result['avg_win_pips']:.2f} pips")
        self.stdout.write(f"Average Loss: {result['avg_loss_pips']:.2f} pips")
        self.stdout.write(f"Profit Factor: {result['profit_factor']:.2f}")

        # Probability ranges analysis
        self.stdout.write(f"\n=== PROBABILITY ANALYSIS ===")
        for prob_range, stats in result['probability_analysis'].items():
            if stats['count'] > 0:
                self.stdout.write(f"{prob_range}: {stats['count']} signals, "
                                f"{stats['accuracy']:.1f}% accuracy, "
                                f"{stats['avg_pips']:.2f} avg pips")

        # Win/Loss distribution
        self.stdout.write(f"\n=== TRADE DISTRIBUTION ===")
        self.stdout.write(f"Wins: {result['wins']} | Losses: {result['losses']}")
        self.stdout.write(f"Largest Win: {result['largest_win']:.1f} pips")
        self.stdout.write(f"Largest Loss: {result['largest_loss']:.1f} pips")

        # Export to CSV if requested
        if export_csv:
            self.stdout.write(f"\n=== EXPORTING TO CSV ===")
            csv_file = ds.export_backtest_to_csv(pair, result['trade_details'])
            self.stdout.write(f"CSV export completed: {csv_file}")