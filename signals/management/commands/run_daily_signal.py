from django.core.management.base import BaseCommand
from signals.models import Signal
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from datetime import datetime, timedelta
import yfinance as yf

class Command(BaseCommand):
    help = 'Generate daily signals for forex pairs'

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true', help='Show what would be done without saving')
        parser.add_argument('--fetch-data', action='store_true', help='Fetch latest data from yfinance before generating signals')

    def handle(self, *args, **options):
        pairs = ['EURUSD', 'XAUUSD']
        data_path = 'data/raw'

        if options['fetch_data']:
            self.fetch_latest_data(pairs, data_path)

        for pair in pairs:
            self.stdout.write(f'Processing {pair}...')
            try:
                # Load data
                file_path = os.path.join(data_path, f'{pair}_Daily.csv')
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()

                # Feature engineering (full set from script)
                df = self.engineer_features(df)

                if len(df) < 30:  # Need more data for features
                    self.stdout.write(f'Not enough data for {pair} after feature engineering')
                    continue

                # Load models
                model_dir = 'models'
                rf_model = joblib.load(os.path.join(model_dir, f'{pair}_rf.joblib'))
                xgb_model = joblib.load(os.path.join(model_dir, f'{pair}_xgb.joblib'))
                scaler = joblib.load(os.path.join(model_dir, f'{pair}_scaler.joblib'))
                calibrator = joblib.load(os.path.join(model_dir, f'{pair}_calibrator.joblib'))

                # Features
                features = [
                    'ret1', 'gapopen', 'sma5', 'sma20', 'ema12', 'ema26', 'rsi14', 'atr14',
                    'bb_width', 'macd', 'macd_signal', 'sma5_20_diff', 'ema12_26_diff',
                    'hl_norm', 'insidebar', 'bullengulf', 'bearengulf', 'pnfdir', 'breakup', 'breakdn'
                ]
                X = df[features].iloc[-1:].values
                X_scaled = scaler.transform(X)

                # Ensemble prediction
                rf_prob = rf_model.predict_proba(X_scaled)[0, 1]
                xgb_prob = xgb_model.predict_proba(X_scaled)[0, 1]
                ensemble_preds = np.column_stack([rf_prob, xgb_prob])
                p_up_calibrated = calibrator.predict_proba(ensemble_preds)[0, 1]

                # Signal logic
                if p_up_calibrated > 0.8:
                    signal = 'bullish'
                elif p_up_calibrated < 0.2:
                    signal = 'bearish'
                else:
                    signal = 'no_signal'

                # Stop loss
                atr = df['atr14'].iloc[-1]
                stop_loss = atr * 0.5

                date = df.index[-1].date()

                if options['dry_run']:
                    self.stdout.write(f'{pair}: {signal} (prob: {p_up_calibrated:.3f}, SL: {stop_loss:.4f}) on {date}')
                else:
                    Signal.objects.create(
                        pair=pair,
                        signal=signal,
                        stop_loss=stop_loss,
                        probability=p_up_calibrated,
                        date=date
                    )
                    self.stdout.write(f'Saved signal for {pair}')

            except Exception as e:
                self.stdout.write(f'Error processing {pair}: {e}')

    def fetch_latest_data(self, pairs, data_path):
        """Fetch latest daily data using yfinance."""
        for pair in pairs:
            self.stdout.write(f'Fetching latest data for {pair}...')
            try:
                # Map forex pairs to Yahoo tickers
                ticker_map = {'EURUSD': 'EURUSD=X', 'XAUUSD': 'GC=F'}
                ticker = ticker_map.get(pair, pair)
                
                # Load existing data
                file_path = os.path.join(data_path, f'{pair}_Daily.csv')
                if os.path.exists(file_path):
                    existing_df = pd.read_csv(file_path)
                    existing_df['date'] = pd.to_datetime(existing_df['date'])
                    last_date = existing_df['date'].max()
                    start_date = last_date + timedelta(days=1)
                else:
                    start_date = datetime.now() - timedelta(days=365*2)  # 2 years of data
                
                # Fetch new data
                data = yf.download(ticker, start=start_date, end=datetime.now() + timedelta(days=1), interval='1d')
                if data.empty:
                    self.stdout.write(f'No new data for {pair}')
                    continue
                
                # Flatten MultiIndex columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # Format to match our CSV structure
                data = data.reset_index()
                data = data.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high', 
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'tickvol'
                })
                data['date'] = pd.to_datetime(data['date']).dt.date
                data['vol'] = 0  # Placeholder
                data['spread'] = 0  # Placeholder
                data = data[['date', 'open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']].dropna()
                
                if data.empty:
                    self.stdout.write(f'No valid new data for {pair}')
                    continue
                
                # Append to existing data
                if os.path.exists(file_path):
                    combined_df = pd.concat([existing_df, data], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset='date', keep='last')
                    combined_df = combined_df.sort_values('date')
                else:
                    combined_df = data
                
                combined_df.to_csv(file_path, index=False)
                self.stdout.write(f'Updated {pair} data: {len(data)} new rows')
                
            except Exception as e:
                self.stdout.write(f'Error fetching data for {pair}: {e}')

    def engineer_features(self, df):
        """Feature engineering from the script."""
        # Basic price features
        df['ret1'] = df['close'].pct_change()
        df['gapopen'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        # Moving averages
        df['sma5'] = df['close'].rolling(5).mean()
        df['sma20'] = df['close'].rolling(20).mean()
        df['ema12'] = df['close'].ewm(span=12).mean()
        df['ema26'] = df['close'].ewm(span=26).mean()

        # RSI
        df['rsi14'] = self.calculate_rsi(df['close'])

        # ATR
        df['atr14'] = self.calculate_atr(df)

        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

        # MACD
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        # Trend diffs
        df['sma5_20_diff'] = df['sma5'] - df['sma20']
        df['ema12_26_diff'] = df['ema12'] - df['ema26']

        # HL normalized
        df['hl_norm'] = (df['high'] - df['low']) / df['close']

        # Price action patterns
        df['insidebar'] = ((df['high'] <= df['high'].shift(1)) & (df['low'] >= df['low'].shift(1))).astype(int)
        df['bullengulf'] = ((df['close'] > df['open']) & (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1))).astype(int)
        df['bearengulf'] = ((df['close'] < df['open']) & (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1))).astype(int)

        # PnF logic
        df['pnfdir'] = np.where(df['close'] > df['close'].shift(1), 1, -1)
        df['pnfdir'] = df['pnfdir'].rolling(5).mean()

        # Breakouts
        df['swinghigh'] = df['high'].rolling(20).max()
        df['swinglow'] = df['low'].rolling(20).min()
        df['breakup'] = (df['close'] > df['swinghigh'].shift(1)).astype(int)
        df['breakdn'] = (df['close'] < df['swinglow'].shift(1)).astype(int)

        df = df.dropna()
        return df

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()