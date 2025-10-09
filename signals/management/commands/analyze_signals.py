from django.core.management.base import BaseCommand
from signals.models import Signal
import pandas as pd

class Command(BaseCommand):
    help = 'Analyze signal performance by probability ranges'

    def handle(self, *args, **options):
        signals = Signal.objects.all().order_by('date')
        if not signals:
            self.stdout.write('No signals to analyze')
            return

        # For simplicity, assume we have next-day data to compare
        # In real implementation, you'd load historical data and compare
        self.stdout.write('Signal Analysis (simplified - needs actual price data for full analysis):')
        
        prob_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        
        for pair in ['EURUSD', 'XAUUSD']:
            pair_signals = signals.filter(pair=pair)
            self.stdout.write(f'\n{pair} Signals:')
            for min_p, max_p in prob_ranges:
                range_signals = pair_signals.filter(probability__gte=min_p, probability__lt=max_p)
                count = range_signals.count()
                if count > 0:
                    bullish = range_signals.filter(signal='bullish').count()
                    bearish = range_signals.filter(signal='bearish').count()
                    self.stdout.write(f'  Prob {min_p:.1f}-{max_p:.1f}: {count} signals ({bullish} bullish, {bearish} bearish)')
                else:
                    self.stdout.write(f'  Prob {min_p:.1f}-{max_p:.1f}: 0 signals')

        self.stdout.write('\nTo get actual win/loss rates, run backtests in daily_forex_signal_system.py')