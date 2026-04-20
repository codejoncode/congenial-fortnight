import pandas as pd
import json
from django.core.management.base import BaseCommand
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Generate today\'s signals for EURUSD and XAUUSD with model probabilities and RR.'

    def add_arguments(self, parser):
        parser.add_argument('--pair', type=str, default=None, help='Currency pair to generate signal for')
        parser.add_argument('--all', action='store_true', help='Generate signals for all pairs')

    def handle(self, *args, **options):
        pairs = ['EURUSD', 'XAUUSD'] if options['all'] or not options['pair'] else [options['pair']]
        today = pd.Timestamp.now().strftime('%Y%m%d')
        signals = []
        for pair in pairs:
            signal = self.generate_signal_for_pair(pair)
            if signal:
                signals.append(signal)
        # Save to signals/signals_YYYYMMDD.json
        out_path = Path(f'signals/signals_{today}.json')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(signals, f, indent=2, default=str)
        self.stdout.write(self.style.SUCCESS(f'Generated signals for {pairs}: {out_path}'))

    def generate_signal_for_pair(self, pair):
        # Load historical data
        csv_path = Path(f'data/{pair}_historical.csv')
        if not csv_path.exists():
            logger.warning(f'No data for {pair}')
            return None
        df = pd.read_csv(csv_path)
        if df.empty:
            logger.warning(f'No data for {pair}')
            return None
        # Prepare features (must match training exactly)
        features = self.prepare_features(df, pair)
        # Model predictions (stub: replace with real model loading)
        rf_pred_proba = 0.78
        xgb_pred_proba = 0.82
        ensemble_confidence = (rf_pred_proba + xgb_pred_proba) / 2
        signal_name = 'RSI_BULLISH_CROSS' if ensemble_confidence > 0.75 else 'BREAKOUT_BEARISH'
        risk_reward_ratio = 2.5
        entry = float(df.iloc[-1]['close'])
        stop_loss = entry - 0.002 if signal_name.startswith('RSI') else entry + 0.002
        take_profit = entry + (abs(entry - stop_loss) * risk_reward_ratio) if signal_name.startswith('RSI') else entry - (abs(entry - stop_loss) * risk_reward_ratio)
        return {
            'pair': pair,
            'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'signal_name': signal_name,
            'rf_pred_proba': rf_pred_proba,
            'xgb_pred_proba': xgb_pred_proba,
            'ensemble_confidence': ensemble_confidence,
            'risk_reward_ratio': risk_reward_ratio,
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

    def prepare_features(self, df, pair):
        # --- COPY YOUR TRAINING FEATURE ENGINEERING HERE ---
        # Example stub:
        df['ret1'] = df['close'].pct_change()
        df['rsi14'] = 50  # stub
        df['macd'] = 0    # stub
        # ...
        feature_columns = ['ret1', 'rsi14', 'macd']
        sample = df[feature_columns].iloc[[-1]]
        return sample
