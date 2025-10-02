from django.core.management.base import BaseCommand
from signals.models import Signal
import pandas as pd
import numpy as np
import os
import joblib
import json
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

    def combine_model_predictions(self, rf_prob, xgb_prob, pair):
        """
        Advanced model combination logic based on agreement and historical performance
        """
        # Get historical performance metrics (simplified - in production, this would be tracked)
        # For now, assume RF performs better on EURUSD, XGB on XAUUSD based on typical patterns
        if pair == 'EURUSD':
            rf_weight = 0.6
            xgb_weight = 0.4
        else:  # XAUUSD
            rf_weight = 0.4
            xgb_weight = 0.6
        
        prob_diff = abs(rf_prob - xgb_prob)
        
        if prob_diff < 0.1:
            # Models agree closely - boost confidence with weighted average
            combined = (rf_prob * rf_weight + xgb_prob * xgb_weight) / (rf_weight + xgb_weight)
            # Add confidence boost for agreement
            agreement_boost = 0.05 * (1 - prob_diff / 0.1)  # Up to 5% boost
            if combined >= 0.5:
                combined = min(0.95, combined + agreement_boost)
            else:
                combined = max(0.05, combined - agreement_boost)
        elif prob_diff > 0.3:
            # Models disagree significantly - follow the more confident model
            rf_confidence = abs(rf_prob - 0.5) * 2
            xgb_confidence = abs(xgb_prob - 0.5) * 2
            
            if rf_confidence > xgb_confidence:
                combined = rf_prob
            else:
                combined = xgb_prob
        else:
            # Moderate disagreement - use weighted average with pair-specific weights
            combined = (rf_prob * rf_weight + xgb_prob * xgb_weight) / (rf_weight + xgb_weight)
        
        return combined

    def handle(self, *args, **options):
        pairs = ['EURUSD', 'XAUUSD']
        data_path = 'data'  # prefer top-level data/ with interval-specific files

        if options['fetch_data']:
            new_data_available = self.fetch_latest_data(pairs, data_path)
        else:
            new_data_available = False

        # Only generate signals if new data is available or explicitly requested
        if not new_data_available and not options.get('force_signals', False):
            self.stdout.write('No new data available. Skipping signal generation.')
            self.stdout.write('Use --fetch-data to check for new data, or --force-signals to generate anyway.')
            return

        signals_generated = False
        for pair in pairs:
            self.stdout.write(f'Processing {pair}...')
            try:
                # Load data - prefer H1 -> H4 -> Daily -> Weekly -> Monthly
                def _find_price_file(pair: str):
                    for interval in ['H1', 'H4', 'Daily', 'Weekly', 'Monthly']:
                        candidate = os.path.join(data_path, f'{pair}_' + interval + '.csv') if interval != 'Daily' else os.path.join(data_path, f'{pair}_Daily.csv')
                        if os.path.exists(candidate):
                            return candidate
                    return None

                file_path = _find_price_file(pair)
                if not file_path:
                    raise FileNotFoundError(f'No data file found for {pair} in {data_path}')

                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()

                # Feature engineering (use same as candle_prediction_system)
                from candle_prediction_system import CandlePredictionSystem
                temp_system = CandlePredictionSystem([pair])
                df = temp_system.engineer_features(df, pair)

                if len(df) < 30:  # Need more data for features
                    self.stdout.write(f'Not enough data for {pair} after feature engineering')
                    continue

                # Load models
                model_dir = 'models'
                rf_model = joblib.load(os.path.join(model_dir, f'{pair}_rf.joblib'))
                xgb_model = joblib.load(os.path.join(model_dir, f'{pair}_xgb.joblib'))
                scaler = joblib.load(os.path.join(model_dir, f'{pair}_scaler.joblib'))
                calibrator = joblib.load(os.path.join(model_dir, f'{pair}_calibrator.joblib'))

                # Use the same features as the training system
                features = temp_system.feature_cols
                X = df[features].iloc[-1:].values
                X_scaled = scaler.transform(X)

                # Ensemble prediction with advanced combination logic
                rf_prob = rf_model.predict_proba(X_scaled)[0, 1]
                xgb_prob = xgb_model.predict_proba(X_scaled)[0, 1]
                
                # Advanced model combination logic
                combined_prob = self.combine_model_predictions(rf_prob, xgb_prob, pair)
                
                # Use calibrated ensemble for final probability
                ensemble_preds = np.column_stack([rf_prob, xgb_prob])
                p_up_calibrated = calibrator.predict_proba(ensemble_preds)[0, 1]
                
                # Use combined probability for signal decision
                final_prob = (combined_prob + p_up_calibrated) / 2

                # Signal logic - always produce a signal based on final probability
                if final_prob >= 0.5:
                    signal = 'bullish'
                else:
                    signal = 'bearish'

                # Stop loss with dynamic adjustment based on confidence
                atr = df['atr_14'].iloc[-1]
                confidence_multiplier = 1.0 + (abs(final_prob - 0.5) * 0.5)  # More confident = wider stop
                stop_loss = atr * 0.5 * confidence_multiplier

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
                    signals_generated = True

            except Exception as e:
                self.stdout.write(f'Error processing {pair}: {e}')

        # Export signals to JSON and send notifications only if signals were generated
        if signals_generated:
            self.export_signals_to_json()
            self.send_notifications()
        elif new_data_available:
            # Send notification that new data was processed but no signals generated
            self.send_data_update_notification()

    def fetch_latest_data(self, pairs, data_path):
        """Fetch latest daily data using yfinance. Returns True if new data was found."""
        new_data_found = False
        
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
                    new_data_found = True  # New file created
                
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
                    
                    # Check if we actually added new rows
                    if len(combined_df) > len(existing_df):
                        new_data_found = True
                        self.stdout.write(f'Updated {pair} data: {len(data)} new rows')
                        try:
                            from scripts.data_metadata import update_metadata
                            update_metadata(file_path)
                        except Exception:
                            pass
                    else:
                        self.stdout.write(f'No new data for {pair}')
                else:
                    data.to_csv(file_path, index=False)
                    self.stdout.write(f'Created {pair} data file with {len(data)} rows')
                    new_data_found = True
                    try:
                        from scripts.data_metadata import update_metadata
                        update_metadata(file_path)
                    except Exception:
                        pass
                
            except Exception as e:
                self.stdout.write(f'Error fetching data for {pair}: {e}')
        
        return new_data_found

    def engineer_features(self, df, pair):
        """Feature engineering with multi-timeframe support."""
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

        # Load weekly data for additional features
        try:
            weekly_file = f'data/raw/{pair}_Weekly.csv'
            if os.path.exists(weekly_file):
                weekly_df = pd.read_csv(weekly_file)
                weekly_df['date'] = pd.to_datetime(weekly_df['date'])

                # Add weekly trend features
                df['date'] = pd.to_datetime(df['date'])

                # Get weekly close for comparison
                weekly_close = []
                for date in df['date']:
                    # Find the most recent weekly close before or on this date
                    weekly_mask = weekly_df['date'] <= date
                    if weekly_mask.any():
                        weekly_close.append(weekly_df.loc[weekly_mask, 'close'].iloc[-1])
                    else:
                        weekly_close.append(df.loc[df['date'] == date, 'close'].iloc[0])

                df['weekly_close'] = weekly_close
                df['weekly_trend'] = (df['close'] - df['weekly_close']) / df['weekly_close']
                df['weekly_momentum'] = df['weekly_close'].pct_change(4)  # 4 weeks momentum
            else:
                df['weekly_trend'] = 0
                df['weekly_momentum'] = 0
        except Exception as e:
            self.stdout.write(f'Warning: Could not load weekly data for {pair}: {e}')
            df['weekly_trend'] = 0
            df['weekly_momentum'] = 0

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

    def export_signals_to_json(self):
        """Export today's signals to JSON file for notifications"""
        try:
            # Get today's signals
            today = datetime.now().date()
            signals = Signal.objects.filter(date=today)

            signal_data = {
                'date': str(today),
                'signals': []
            }

            for signal in signals:
                # Calculate entry price (current close for simplicity)
                entry_price = self.get_current_price(signal.pair)

                signal_dict = {
                    'pair': signal.pair,
                    'signal': signal.signal,
                    'probability': float(signal.probability),
                    'stop_loss': float(signal.stop_loss),
                    'entry_price': entry_price,
                    'date': str(signal.date)
                }
                signal_data['signals'].append(signal_dict)

            # Save to output directory
            os.makedirs('output', exist_ok=True)
            with open('output/daily_signals.json', 'w') as f:
                json.dump(signal_data, f, indent=2)

            self.stdout.write(f'Exported {len(signal_data["signals"])} signals to JSON')

        except Exception as e:
            self.stdout.write(f'Error exporting signals to JSON: {e}')

    def get_current_price(self, pair):
        """Get current price for entry calculation"""
        try:
            # Load latest data
            file_path = f'data/raw/{pair}_Daily.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                return float(df['close'].iloc[-1])
            return 0.0
        except:
            return 0.0

    def send_notifications(self):
        """Send signal notifications to configured recipients"""
        try:
            from notification_system import NotificationSystem
            
            notifier = NotificationSystem()
            
            # Get today's signals
            today = datetime.now().date()
            signals = Signal.objects.filter(date=today)
            
            if signals:
                signal_list = []
                for signal in signals:
                    entry_price = self.get_current_price(signal.pair)
                    signal_list.append({
                        'pair': signal.pair,
                        'signal': signal.signal,
                        'probability': float(signal.probability),
                        'stop_loss': float(signal.stop_loss),
                        'entry_price': entry_price,
                        'date': str(signal.date)
                    })
                
                recipients = []
                email = os.getenv('NOTIFICATION_EMAIL')
                sms = os.getenv('NOTIFICATION_SMS')
                if email:
                    recipients.append(email)
                if sms:
                    recipients.append(sms)
                
                if recipients:
                    notifier.send_signal_notification(signal_list, recipients)
                    self.stdout.write(f'Sent notifications to {len(recipients)} recipients')
                else:
                    self.stdout.write('No notification recipients configured')
            else:
                self.stdout.write('No signals to notify about')
                
        except Exception as e:
            self.stdout.write(f'Error sending notifications: {e}')

    def send_data_update_notification(self):
        """Send notification that new data was processed but no signals generated"""
        try:
            from notification_system import NotificationSystem
            
            notifier = NotificationSystem()
            
            message = f"📊 Forex Data Update - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            message += "New market data has been processed and models updated.\n"
            message += "No new trading signals generated at this time.\n\n"
            message += "Models are ready for the next trading session."
            
            recipients = []
            email = os.getenv('NOTIFICATION_EMAIL')
            sms = os.getenv('NOTIFICATION_SMS')
            if email:
                recipients.append(email)
            if sms:
                recipients.append(sms)
            
            if recipients:
                notifier.send_notification(
                    subject="📊 Forex Data Update - No New Signals",
                    message=message,
                    recipients=recipients
                )
                self.stdout.write('Sent data update notification')
            else:
                self.stdout.write('No notification recipients configured')
                
        except Exception as e:
            self.stdout.write(f'Error sending data update notification: {e}')