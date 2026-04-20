from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from .models import Signal
from .serializers import SignalSerializer
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import logging
from trading_system import TradingDataCollector, TradingStrategies

# Configure logging
logger = logging.getLogger(__name__)

# Helper function to get current price from latest data
def get_current_price(pair):
    """Get current/latest price for a pair from CSV data"""
    try:
        # Try to find price file
        for interval in ['H1', 'H4', 'Daily']:
            data_path = f'data/{pair}_{interval}.csv' if interval != 'Daily' else f'data/{pair}_Daily.csv'
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                if 'close' in df.columns and len(df) > 0:
                    return float(df.iloc[-1]['close'])
        
        # Fallback defaults for common pairs
        defaults = {
            'EURUSD': 1.0850,
            'XAUUSD': 2650.00,
            'GBPUSD': 1.2700,
            'USDJPY': 149.50
        }
        return defaults.get(pair, 1.0)
    except Exception as e:
        logger.error(f"Error getting current price for {pair}: {e}")
        return 1.0  # Safe fallback

class SignalViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Signal.objects.all().order_by('-date')
    serializer_class = SignalSerializer
    permission_classes = [AllowAny]

@api_view(['GET'])
@permission_classes([AllowAny])
def backtest_results(request):
    pair = request.GET.get('pair', 'EURUSD')
    days = int(request.GET.get('days', 30))

    try:
        # Import the enhanced backtest system
        import sys
        import os
        import logging
        sys.path.append(os.getcwd())
        from daily_forex_signal_system import DailyForexSignal

        # Log backtest start
        print(f"Starting backtest for {pair} over {days} days...")
        logging.info(f"Backtest started: {pair}, {days} days")

        # Run enhanced backtest
        ds = DailyForexSignal()
        result = ds.backtest_last_n_days_enhanced(pair, n=days)

        # Log backtest completion
        print(f"Backtest completed for {pair}. Results: {result['total_signals']} signals, {result['wins']} wins, {result['losses']} losses")
        logging.info(f"Backtest completed: {pair}, {result['total_signals']} signals")

        # Format response for frontend
        response_data = {
            'status': 'completed',
            'message': f'Backtest completed successfully for {pair} over {days} days',
            'pair': result['pair'],
            'days': result['period_days'],
            'overall_accuracy': result['accuracy'] / 100,  # Convert to decimal for frontend
            'total_signals': result['total_signals'],
            'wins': result['wins'],
            'losses': result['losses'],
            'total_pips_won': result['total_pips_won'],
            'total_pips_lost': result['total_pips_lost'],
            'net_pips': result['net_pips'],
            'avg_win_pips': result['avg_win_pips'],
            'avg_loss_pips': result['avg_loss_pips'],
            'profit_factor': result['profit_factor'],
            'largest_win': result['largest_win'],
            'largest_loss': result['largest_loss'],
            'probability_bins': {}
        }

        # Convert probability analysis to frontend format
        for prob_range, stats in result['probability_analysis'].items():
            response_data['probability_bins'][prob_range] = {
                'total': stats['count'],
                'correct': int(stats['count'] * stats['accuracy'] / 100) if stats['count'] > 0 else 0,
                'accuracy': stats['accuracy'] / 100  # Convert to decimal
            }

        # Add recent trade details (last 10)
        response_data['recent_results'] = []
        for trade in result['trade_details'][-10:]:  # Last 10 trades
            response_data['recent_results'].append({
                'date': trade['date'],
                'signal': trade['signal'],
                'actual': 'bullish' if trade['pips'] > 0 else 'bearish',
                'correct': trade['profitable'],
                'probability': trade['probability'],
                'pips': trade['pips']
            })

        return Response(response_data)

    except Exception as e:
        print(f"Backtest failed for {pair}: {str(e)}")
        logging.error(f"Backtest failed: {pair}, {days} days - {str(e)}")
        return Response({
            'status': 'error',
            'message': f'Backtest failed for {pair}: {str(e)}',
            'error': str(e)
        })

@api_view(['GET'])
@permission_classes([AllowAny])
def download_backtest_csv(request):
    """Download backtest results as CSV file"""
    pair = request.GET.get('pair', 'EURUSD')
    days = int(request.GET.get('days', 30))

    try:
        from daily_forex_signal_system import DailyForexSignal
        import io
        from django.http import HttpResponse

        ds = DailyForexSignal()
        result = ds.backtest_last_n_days_enhanced(pair, n=days)

        if 'trade_details' in result and result['trade_details']:
            df = pd.DataFrame(result['trade_details'])
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()

            # Create HTTP response with CSV
            response = HttpResponse(csv_content, content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="backtest_{pair}_{days}days_{datetime.now().strftime("%Y%m%d")}.csv"'
            return response
        else:
            return Response({'error': 'No trade data available'}, status=404)

    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
@permission_classes([AllowAny])
def get_historical_data(request):
    """Get historical price data for charting"""
    pair = request.GET.get('pair', 'EURUSD')
    days = int(request.GET.get('days', 30))

    try:
        # Load historical data - prefer interval files in data/
        import os
        def _find_price_file(pair: str):
            for interval in ['H1', 'H4', 'Daily', 'Weekly', 'Monthly']:
                candidate = f'data/{pair}_' + interval + '.csv' if interval != 'Daily' else f'data/{pair}_Daily.csv'
                if os.path.exists(candidate):
                    return candidate
            return None

        data_path = _find_price_file(pair)
        if not data_path or not os.path.exists(data_path):
            return Response({'error': 'Data file not found'}, status=404)

        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Get last N days
        recent_data = df.tail(days)
        
        # Format for frontend
        chart_data = []
        for _, row in recent_data.iterrows():
            chart_data.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row.get('tickvol', 0))
            })

        return Response(chart_data)

    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """Simple health check endpoint"""
    return Response({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'trading_system'
    })


@api_view(['GET'])
@permission_classes([AllowAny])
def system_health(request):
    """Full system health snapshot — data, models, signals, positions."""
    from signals.health import get_system_health
    return Response(get_system_health())


@api_view(['GET'])
@permission_classes([AllowAny])
def signal_decision(request, pair=None):
    """
    Run the decision engine on the latest DB signal for a pair.

    Query params:
        open_positions: int (default 0)
        daily_pnl:      float (default 0.0)
    """
    from signals.decision_engine import evaluate

    pairs_to_check = [pair.upper()] if pair else ['EURUSD', 'XAUUSD']
    open_pos   = int(request.GET.get('open_positions', 0))
    daily_pnl  = float(request.GET.get('daily_pnl', 0.0))
    balance    = float(request.GET.get('balance', 500.0))

    results = {}
    for p in pairs_to_check:
        sig_obj = Signal.objects.filter(pair=p).order_by('-date', '-id').first()
        if not sig_obj:
            results[p] = {'action': 'SKIP', 'summary': 'No signal in database for this pair.', 'reasons': [], 'score': 0}
            continue

        sig_dict = {
            'pair':        p,
            'signal':      sig_obj.signal,
            'probability': float(sig_obj.probability),
            'confidence':  float(getattr(sig_obj, 'confidence', 0) or 0),
            'entry_price': float(sig_obj.entry_price) if sig_obj.entry_price else None,
            'entry':       float(sig_obj.entry_price) if sig_obj.entry_price else None,
            'stop_loss':   float(sig_obj.stop_loss)   if sig_obj.stop_loss   else None,
            'take_profit': float(sig_obj.take_profit) if sig_obj.take_profit else None,
            'risk_reward': float(getattr(sig_obj, 'risk_reward', 0) or 0),
            'atr':         float(getattr(sig_obj, 'atr', 0) or 0),
            'date':        sig_obj.date.isoformat(),
        }

        decision = evaluate(sig_dict, open_positions=open_pos, daily_pnl_usd=daily_pnl, account_balance=balance)
        results[p] = {**decision, 'signal': sig_dict}

    return Response(results)

@api_view(['GET'])
@permission_classes([AllowAny])
def unified_signals(request):
    """
    Get unified signals from both ML Pip-Based and Harmonic Pattern systems
    
    Query params:
        pair: Currency pair (default: EURUSD)
        mode: parallel|confluence|weighted (default: parallel)
    """
    pair = request.GET.get('pair', 'EURUSD')
    mode = request.GET.get('mode', 'parallel')
    
    try:
        import sys
        import os
        from pathlib import Path
        
        # Add scripts to path
        sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
        
        from unified_signal_service import UnifiedSignalService
        import joblib
        
        # Load historical data
        data_file = f'data/{pair}_H1.csv'
        if not os.path.exists(data_file):
            return Response({
                'error': f'Data file not found: {data_file}',
                'pair': pair
            }, status=404)
        
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').tail(5000)  # Last 5000 bars
        
        # Load ML model
        model_file = f'models/{pair}_ensemble.joblib'
        if not os.path.exists(model_file):
            return Response({
                'error': f'Model file not found: {model_file}',
                'pair': pair
            }, status=404)
        
        ml_model = joblib.load(model_file)
        
        # Generate unified signals
        service = UnifiedSignalService(mode=mode)
        signals = service.generate_unified_signals(pair, df, ml_model)
        
        return Response(signals)
        
    except Exception as e:
        logger.error(f"Error generating unified signals: {e}", exc_info=True)
        return Response({
            'error': str(e),
            'pair': pair,
            'mode': mode
        }, status=500)

@api_view(['GET'])
@permission_classes([AllowAny])
def get_signals(request, pair=None):
    """Get current model-based trading signals for one or more pairs using pip-based risk rules.

    If `pair` is provided, returns a list with signals for that pair (if any).
    If no `pair` is provided, returns signals for all supported pairs.
    """
    try:
        from scripts.pip_based_signal_system import PipBasedSignalSystem
        import joblib
        from pathlib import Path

        data_dir = Path('data')
        models_dir = Path('models')

        # Pairs we actively support
        supported_pairs = ['EURUSD', 'XAUUSD']
        if pair:
            supported_pairs = [pair]

        pip_system = PipBasedSignalSystem()
        results = []

        for p in supported_pairs:
            # Load latest H1 data
            data_file = data_dir / f'{p}_H1.csv'
            if not data_file.exists():
                logger.warning(f"No data file found for {p}: {data_file}")
                continue

            df = pd.read_csv(data_file)
            if df.empty:
                logger.warning(f"Data file for {p} is empty")
                continue

            # Ensure timestamp and sort
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').tail(1000)

            # Use real SignalEngine prediction for direction + confidence.
            # Falls back to last-candle heuristic if models are not yet trained.
            last_row = df.iloc[-1]
            try:
                from signals.signal_engine import SignalEngine
                _engine = SignalEngine()
                if _engine.models_exist(p):
                    _df_indexed = df.set_index('timestamp').sort_index()
                    _pred = _engine.predict(p, _df_indexed)
                    direction = 'long' if _pred['signal'] == 'bullish' else 'short'
                    model_prediction = {
                        'direction': direction,
                        'confidence': round(_pred['probability'], 4),
                        'engine_signal': _pred['signal'],
                        'engine_prob': _pred['probability'],
                    }
                else:
                    direction = 'long' if last_row['close'] > last_row['open'] else 'short'
                    model_prediction = {'direction': direction, 'confidence': 0.5}
            except Exception as _e:
                logger.warning(f'SignalEngine predict failed for {p}: {_e}')
                direction = 'long' if last_row['close'] > last_row['open'] else 'short'
                model_prediction = {'direction': direction, 'confidence': 0.5}

            # Use pip-based system to detect a quality setup
            signal = pip_system.detect_quality_setup(
                df.set_index('timestamp'),
                p,
                model_prediction
            )

            if signal.get('signal') is None:
                logger.info(f"No quality setup for {p} at this time")
                continue

            # Enforce spread-aware stop: risk must exceed typical spread * 2
            typical_spread = pip_system.typical_spreads.get(p, 1.0)
            if signal['risk_pips'] < typical_spread * 2:
                logger.info(
                    f"Rejected {p} signal due to tight stop: risk {signal['risk_pips']:.1f} pips, "
                    f"spread {typical_spread} pips"
                )
                continue

            results.append({
                'pair': p,
                'signal': 'bullish' if signal['signal'] == 'long' else 'bearish',
                'probability': signal['confidence'],
                'entry': signal['entry'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'risk_pips': signal['risk_pips'],
                'reward_pips': signal['reward_pips'],
                'risk_reward_ratio': signal['risk_reward_ratio'],
                'date': signal.get('timestamp').strftime('%Y-%m-%d') if signal.get('timestamp') is not None else df['timestamp'].iloc[-1].strftime('%Y-%m-%d'),
                'timestamp': signal.get('timestamp').isoformat() if signal.get('timestamp') is not None else df['timestamp'].iloc[-1].isoformat(),
                'setup_quality': signal['setup_quality'],
                'quality_score': signal['quality_score'],
                'reasoning': signal['reasoning']
            })

        return Response(results)

    except Exception as e:
        logger.error(f"Error getting signals: {str(e)}", exc_info=True)
        return Response({
            'error': f'Failed to get signals: {str(e)}'
        }, status=500)

@api_view(['GET'])
@permission_classes([AllowAny])
def trading_backtest(request, pair):
    """Run backtest for trading strategies on a specific pair"""
    try:
        # Initialize trading system
        data_collector = TradingDataCollector()
        strategies = TradingStrategies(data_collector)

        # Load data
        data = data_collector.collect_all_data()

        if pair not in data or data[pair].empty:
            return Response({
                'error': f'No data available for {pair}'
            }, status=404)

        df = data[pair]

        # Run master signal system
        signals = strategies.master_signal_system(df)

        # Simple backtest calculation
        returns = df['Close'].pct_change()
        signal_returns = signals.shift(1) * returns  # Shift signals to avoid lookahead bias

        total_trades = signals.abs().sum()
        winning_trades = ((signals.shift(1) * returns) > 0).sum()
        accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        total_return = signal_returns.sum()
        sharpe_ratio = signal_returns.mean() / signal_returns.std() * np.sqrt(252) if signal_returns.std() > 0 else 0

        return Response({
            'pair': pair,
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'accuracy': round(accuracy, 2),
            'total_return': round(total_return, 4),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'data_points': len(df)
        })

    except Exception as e:
        logger.error(f"Error running backtest for {pair}: {str(e)}")
        return Response({
            'error': f'Failed to run backtest for {pair}: {str(e)}'
        }, status=500)

@api_view(['GET'])
@permission_classes([AllowAny])
def data_status(request):
    """Get status of data collection"""
    try:
        data_collector = TradingDataCollector()

        # Check data files
        data_files = {
            'EURUSD_H1': os.path.exists('data/EURUSD_H1.csv'),
            'EURUSD_Monthly': os.path.exists('data/EURUSD_Monthly.csv'),
            'XAUUSD_H1': os.path.exists('data/XAUUSD_H1.csv'),
            'XAUUSD_Monthly': os.path.exists('data/XAUUSD_Monthly.csv')
        }

        # Check API call counts
        api_status = {
            'fred_calls': data_collector.api_calls['fred'],
            'finnhub_calls': data_collector.api_calls['finnhub'],
            'fmp_calls': data_collector.api_calls['fmp'],
            'yahoo_calls': data_collector.api_calls['yahoo'],
            'ecb_calls': data_collector.api_calls['ecb']
        }

        return Response({
            'data_files': data_files,
            'api_calls': api_status,
            'last_updated': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting data status: {str(e)}")
        return Response({
            'error': f'Failed to get data status: {str(e)}'
        }, status=500)

@api_view(['POST'])
@permission_classes([AllowAny])
def update_data_old(request):
    """Old update_data - renamed to avoid conflict"""
    try:
        data_collector = TradingDataCollector()

        # Collect all data
        data = data_collector.collect_all_data()

        # Save data (simplified - in real implementation, save to files)
        updated_pairs = list(data.keys())
        total_records = sum(len(df) for df in data.values())

        return Response({
            'status': 'success',
            'message': f'Updated data for {len(updated_pairs)} pairs',
            'pairs': updated_pairs,
            'total_records': total_records,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error updating data: {str(e)}")
        return Response({
            'error': f'Failed to update data: {str(e)}'
        }, status=500)


@api_view(['POST'])
@permission_classes([AllowAny])
def update_data(request):
    """
    Update trading data using FREE APIs with multiple fallbacks.
    Tries sources in order: Yahoo Finance -> Twelve Data -> Alpha Vantage
    Ensures data is ALWAYS fetched from at least one source.
    
    Saves data to CSV files in data/ directory.
    """
    try:
        import requests
        import time
        from pathlib import Path
        import yfinance as yf
        
        data_dir = Path('data')
        data_dir.mkdir(parents=True, exist_ok=True)
        updated_files = []
        errors = []
        
        alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')
        twelve_data_key = os.getenv('TWELVEDATA_API_KEY', '')  # Note: TWELVEDATA not TWELVE_DATA
        
        def fetch_yahoo_finance(symbol, period='60d', interval='1h'):
            """Fetch data from Yahoo Finance (Free, unlimited, but sometimes rate-limited)"""
            try:
                logger.info(f"Fetching {symbol} from Yahoo Finance...")
                df = yf.download(symbol, period=period, interval=interval, progress=False)
                
                if df.empty:
                    logger.warning(f"Yahoo Finance returned empty data for {symbol}")
                    return None
                
                # Prepare data
                df = df.reset_index()
                
                # Handle column names (Datetime or Date)
                timestamp_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
                df['timestamp'] = pd.to_datetime(df[timestamp_col])
                df['date'] = df['timestamp'].dt.date
                df['time'] = df['timestamp'].dt.time
                
                # Rename OHLCV columns to lowercase
                df = df.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low',
                    'Close': 'close', 'Volume': 'volume'
                })
                
                # Add required columns
                df['id'] = range(1, len(df) + 1)
                df['spread'] = 2
                
                # Select and order columns
                df = df[['id', 'timestamp', 'date', 'time', 'open', 'high', 'low', 'close', 'volume', 'spread']]
                
                logger.info(f"✅ Yahoo Finance: Fetched {len(df)} records")
                return df
                
            except Exception as e:
                logger.error(f"❌ Yahoo Finance failed: {e}")
                return None
        
        def fetch_alpha_vantage_forex(from_currency, to_currency, interval='60min'):
            """Fetch forex data from Alpha Vantage (Free: 500 calls/day, 5 calls/min)"""
            if not alpha_vantage_key:
                logger.warning("Alpha Vantage API key not configured")
                return None
                
            try:
                url = 'https://www.alphavantage.co/query'
                params = {
                    'function': 'FX_DAILY',  # Use daily instead of intraday (free tier)
                    'from_symbol': from_currency,
                    'to_symbol': to_currency,
                    'outputsize': 'full',
                    'apikey': alpha_vantage_key
                }
                
                logger.info(f"Fetching {from_currency}/{to_currency} from Alpha Vantage...")
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if 'Note' in data:
                    logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                    return None
                
                if 'Information' in data:
                    logger.warning(f"Alpha Vantage info: {data['Information']}")
                    return None
                
                time_series_key = 'Time Series FX (Daily)'
                if time_series_key in data:
                    df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()
                    
                    # Rename columns
                    df.columns = ['open', 'high', 'low', 'close']
                    df = df.astype(float)
                    
                    # Add required columns
                    df = df.reset_index()
                    df = df.rename(columns={'index': 'timestamp'})
                    df['date'] = df['timestamp'].dt.date
                    df['time'] = df['timestamp'].dt.time
                    df['id'] = range(1, len(df) + 1)
                    df['volume'] = 0
                    df['spread'] = 2
                    
                    # Reorder columns
                    df = df[['id', 'timestamp', 'date', 'time', 'open', 'high', 'low', 'close', 'volume', 'spread']]
                    
                    logger.info(f"✅ Alpha Vantage: Fetched {len(df)} records")
                    return df
                else:
                    logger.error(f"Alpha Vantage unexpected response: {data}")
                    return None
                    
            except Exception as e:
                logger.error(f"❌ Alpha Vantage failed: {e}")
                return None
        
        def fetch_twelve_data(symbol, interval='1h'):
            """Fetch data from Twelve Data (Free: 800 calls/day, works with 'demo' key)"""
            try:
                url = 'https://api.twelvedata.com/time_series'
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'outputsize': 5000,  # Max for free tier
                    'apikey': twelve_data_key
                }
                
                logger.info(f"Fetching {symbol} from Twelve Data...")
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if 'values' in data and data['values']:
                    df = pd.DataFrame(data['values'])
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df = df.sort_values('datetime')
                    
                    # Convert to numeric
                    for col in ['open', 'high', 'low', 'close']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Handle volume column properly
                    if 'volume' in df.columns:
                        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
                    else:
                        df['volume'] = 0
                    
                    df = df.rename(columns={'datetime': 'timestamp'})
                    df['date'] = df['timestamp'].dt.date
                    df['time'] = df['timestamp'].dt.time
                    df['id'] = range(1, len(df) + 1)
                    df['spread'] = 2
                    
                    df = df[['id', 'timestamp', 'date', 'time', 'open', 'high', 'low', 'close', 'volume', 'spread']]
                    
                    logger.info(f"✅ Twelve Data: Fetched {len(df)} records")
                    return df
                else:
                    logger.error(f"Twelve Data unexpected response: {data}")
                    return None
                    
            except Exception as e:
                logger.error(f"❌ Twelve Data failed: {e}")
                return None
        
        def create_daily_from_hourly(df_hourly):
            """Resample hourly data to daily"""
            try:
                df = df_hourly.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                
                daily = df.resample('D').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                daily = daily.reset_index()
                daily['date'] = daily['timestamp'].dt.date
                daily['time'] = daily['timestamp'].dt.time
                daily['id'] = range(1, len(daily) + 1)
                daily['spread'] = 2
                
                return daily[['id', 'timestamp', 'date', 'time', 'open', 'high', 'low', 'close', 'volume', 'spread']]
            except Exception as e:
                logger.error(f"Error creating daily data: {e}")
                return None
        
        def create_all_timeframes_from_h1(df_hourly, pair_name):
            """Create H4, Daily, Weekly, Monthly from H1 data and save all"""
            timeframe_files = []
            
            try:
                df = df_hourly.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df_indexed = df.set_index('timestamp')
                
                def format_resampled(resampled_df):
                    resampled_df = resampled_df.reset_index()
                    resampled_df['date'] = resampled_df['timestamp'].dt.date
                    resampled_df['time'] = resampled_df['timestamp'].dt.time
                    resampled_df['id'] = range(1, len(resampled_df) + 1)
                    resampled_df['spread'] = 2
                    return resampled_df[['id', 'timestamp', 'date', 'time', 'open', 'high', 'low', 'close', 'volume', 'spread']]
                
                timeframes = {
                    'H4': '4h',
                    'Daily': 'D',
                    'Weekly': 'W',
                    'Monthly': 'ME'
                }
                
                for tf_name, resample_rule in timeframes.items():
                    resampled = df_indexed.resample(resample_rule).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                    
                    tf_df = format_resampled(resampled)
                    tf_file = data_dir / f'{pair_name}_{tf_name}.csv'
                    tf_df.to_csv(tf_file, index=False)
                    timeframe_files.append(str(tf_file.name))
                    logger.info(f"✅ Saved {tf_file.name} with {len(tf_df)} rows")
                
                return timeframe_files
                
            except Exception as e:
                logger.error(f"Error creating timeframes for {pair_name}: {e}")
                return []
        
        # Update EURUSD - Try all sources until one works
        logger.info("=" * 60)
        logger.info("Updating EURUSD...")
        eurusd_df = None
        sources_tried = []
        
        # Source 1: Yahoo Finance (fastest, no rate limits)
        try:
            eurusd_df = fetch_yahoo_finance('EURUSD=X', period='60d', interval='1h')
            sources_tried.append('Yahoo Finance')
        except Exception as e:
            logger.error(f"Yahoo Finance exception for EURUSD: {e}")
            eurusd_df = None
            sources_tried.append('Yahoo Finance')
        
        # Source 2: Twelve Data (if Yahoo failed)
        if eurusd_df is None or eurusd_df.empty:
            logger.info("Yahoo failed, trying Twelve Data...")
            try:
                eurusd_df = fetch_twelve_data('EUR/USD', '1h')
                sources_tried.append('Twelve Data')
                time.sleep(8)  # Rate limit: 8 calls/min
            except Exception as e:
                logger.error(f"Twelve Data exception for EURUSD: {e}")
                eurusd_df = None
        
        # Source 3: Alpha Vantage (last resort)
        if eurusd_df is None or eurusd_df.empty:
            logger.info("Twelve Data failed, trying Alpha Vantage...")
            try:
                eurusd_df = fetch_alpha_vantage_forex('EUR', 'USD')
                sources_tried.append('Alpha Vantage')
                time.sleep(12)  # Rate limit: 5 calls/min
            except Exception as e:
                logger.error(f"Alpha Vantage exception for EURUSD: {e}")
                eurusd_df = None
        
        if eurusd_df is not None and not eurusd_df.empty:
            # Save H1 data
            h1_file = data_dir / 'EURUSD_H1.csv'
            eurusd_df.to_csv(h1_file, index=False)
            updated_files.append(str(h1_file.name))
            latest_date = eurusd_df['timestamp'].max()
            logger.info(f"✅ EURUSD: Saved {h1_file.name} with {len(eurusd_df)} rows (Latest: {latest_date})")
            logger.info(f"   Data source: {sources_tried[-1]}")
            
            # Create and save all timeframes (H4, Daily, Weekly, Monthly)
            tf_files = create_all_timeframes_from_h1(eurusd_df, 'EURUSD')
            updated_files.extend(tf_files)
        else:
            error_msg = f"EURUSD: Failed from all sources: {', '.join(sources_tried)}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        # Update XAUUSD (Gold) - Try all sources until one works
        logger.info("=" * 60)
        logger.info("Updating XAUUSD...")
        xauusd_df = None
        sources_tried = []
        
        # Source 1: Yahoo Finance (Gold futures as proxy)
        try:
            xauusd_df = fetch_yahoo_finance('GC=F', period='60d', interval='1h')
            sources_tried.append('Yahoo Finance')
        except Exception as e:
            logger.error(f"Yahoo Finance exception for XAUUSD: {e}")
            xauusd_df = None
            sources_tried.append('Yahoo Finance')
        
        # Source 2: Twelve Data
        if xauusd_df is None or xauusd_df.empty:
            logger.info("Yahoo failed, trying Twelve Data...")
            try:
                xauusd_df = fetch_twelve_data('XAU/USD', '1h')
                sources_tried.append('Twelve Data')
                time.sleep(8)
            except Exception as e:
                logger.error(f"Twelve Data exception for XAUUSD: {e}")
                xauusd_df = None
        
        # Source 3: Alpha Vantage
        if xauusd_df is None or xauusd_df.empty:
            logger.info("Twelve Data failed, trying Alpha Vantage...")
            try:
                xauusd_df = fetch_alpha_vantage_forex('XAU', 'USD')
                sources_tried.append('Alpha Vantage')
                time.sleep(12)
            except Exception as e:
                logger.error(f"Alpha Vantage exception for XAUUSD: {e}")
                xauusd_df = None
        
        if xauusd_df is not None and not xauusd_df.empty:
            # Save H1 data
            h1_file = data_dir / 'XAUUSD_H1.csv'
            xauusd_df.to_csv(h1_file, index=False)
            updated_files.append(str(h1_file.name))
            latest_date = xauusd_df['timestamp'].max()
            logger.info(f"✅ XAUUSD: Saved {h1_file.name} with {len(xauusd_df)} rows (Latest: {latest_date})")
            logger.info(f"   Data source: {sources_tried[-1]}")
            
            # Create and save all timeframes (H4, Daily, Weekly, Monthly)
            tf_files = create_all_timeframes_from_h1(xauusd_df, 'XAUUSD')
            updated_files.extend(tf_files)
        else:
            error_msg = f"XAUUSD: Failed from all sources: {', '.join(sources_tried)}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        logger.info("=" * 60)
        
        # Build response
        if updated_files:
            return Response({
                'status': 'success',
                'message': f'Successfully updated {len(updated_files)} files',
                'files': updated_files,
                'pairs': ['EURUSD', 'XAUUSD'],
                'errors': errors if errors else None,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return Response({
                'status': 'error',
                'message': 'Failed to update any data',
                'errors': errors,
                'timestamp': datetime.now().isoformat()
            }, status=500)
            
    except Exception as e:
        logger.error(f"Critical error in update_data: {str(e)}", exc_info=True)
        return Response({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }, status=500)


@api_view(['POST'])
@permission_classes([AllowAny])
def generate_signals(request):
    """
    Trigger signal generation for all pairs via management command, then return
    the latest signals from the database.  Signals are always returned even when
    the command skips generation (e.g. no new market data today).
    """
    from django.core.management import call_command
    from io import StringIO

    output = ''
    generation_attempted = False

    try:
        out = StringIO()
        call_command('run_daily_signal', '--force', '--fetch-data', stdout=out)
        output = out.getvalue()
        generation_attempted = True
    except Exception as cmd_err:
        logger.warning(f"run_daily_signal command error (non-fatal): {cmd_err}")
        output = str(cmd_err)

    # Always return the latest DB signals regardless of whether generation succeeded
    def _signal_dict(s):
        return {
            'id': s.id,
            'pair': s.pair,
            'signal': s.signal,
            'probability': float(s.probability),
            'stop_loss': float(s.stop_loss) if s.stop_loss is not None else None,
            'date': s.date.isoformat(),
            'created_at': s.created_at.isoformat() if hasattr(s, 'created_at') and s.created_at else None,
        }

    # One latest signal per pair
    signals_data = []
    for pair in ['EURUSD', 'XAUUSD']:
        qs = Signal.objects.filter(pair=pair).order_by('-date', '-id').first()
        if qs:
            signals_data.append(_signal_dict(qs))

    return Response({
        'status': 'success',
        'message': f'Returning {len(signals_data)} signal(s)',
        'generation_attempted': generation_attempted,
        'signals': signals_data,
        'output': output,
        'timestamp': datetime.now().isoformat(),
    })


@api_view(['GET'])
@permission_classes([AllowAny])
def get_holloway(request, pair):
    """Return per-timeframe Holloway summary (bull/bear counts + averages) for a pair."""
    try:
        data_dir = os.path.join(os.getcwd(), 'data')
        timeframes = ['daily', 'h4', 'h1', 'weekly', 'monthly']
        out = {}
        for tf in timeframes:
            fname = os.path.join(data_dir, f"{pair}_{tf}_complete_holloway.csv")
            if not os.path.exists(fname):
                out[tf] = None
                continue
            df = pd.read_csv(fname)
            df.columns = [c.lower() for c in df.columns]
            if len(df) == 0:
                out[tf] = None
                continue
            latest = df.iloc[-1]
            def safe(col):
                return float(latest[col]) if col in df.columns and pd.notna(latest[col]) else None

            out[tf] = {
                'bull_count': safe('bull_count'),
                'bear_count': safe('bear_count'),
                'bully': safe('bully'),
                'beary': safe('beary'),
                'holloway_bull_signals': int(df['holloway_bull_signal'].sum()) if 'holloway_bull_signal' in df.columns else 0,
                'holloway_bear_signals': int(df['holloway_bear_signal'].sum()) if 'holloway_bear_signal' in df.columns else 0,
                'data_points': len(df),
                'filepath': fname
            }

        # also include merged latest features if available
        merged = os.path.join(data_dir, f"{pair}_latest_multi_timeframe_features.csv")
        if os.path.exists(merged):
            mdf = pd.read_csv(merged)
            mdf.columns = [c.lower() for c in mdf.columns]
            out['latest_merged'] = mdf.to_dict(orient='records')[0] if len(mdf) > 0 else {}

        return Response({'pair': pair, 'holloway': out})
    except Exception as e:
        logger.error(f"Error getting holloway for {pair}: {str(e)}")
        return Response({'error': str(e)}, status=500)


@api_view(['POST'])
@permission_classes([AllowAny])
def execute_paper_trade(request):
    """
    Execute a paper trade from a signal
    
    POST body:
        pair: Currency pair (e.g., 'EURUSD')
        signal: 'bullish' or 'bearish'
        stop_loss: Stop loss price
        probability: Signal confidence (0-1)
        lot_size: Position size in lots (default: 0.1)
    
    Returns:
        Trade execution details
    """
    try:
        # Import paper trading engine
        from paper_trading.engine import PaperTradingEngine
        
        # Get request data
        pair = request.data.get('pair')
        signal = request.data.get('signal')
        stop_loss = request.data.get('stop_loss')
        probability = request.data.get('probability', 0.5)
        lot_size = float(request.data.get('lot_size', 0.1))
        
        # Validate inputs
        if not pair or not signal:
            return Response({
                'success': False,
                'error': 'Missing required fields: pair and signal'
            }, status=400)
        
        if signal not in ['bullish', 'bearish']:
            return Response({
                'success': False,
                'error': 'Signal must be bullish or bearish'
            }, status=400)
        
        # Get current price
        current_price = get_current_price(pair)
        
        # Calculate take profit based on signal
        if signal == 'bullish':
            order_type = 'buy'
            # TP is 1.5x the risk (SL distance)
            sl_distance = current_price - float(stop_loss) if stop_loss else current_price * 0.02
            take_profit = current_price + (sl_distance * 1.5)
        else:
            order_type = 'sell'
            sl_distance = float(stop_loss) - current_price if stop_loss else current_price * 0.02
            take_profit = current_price - (sl_distance * 1.5)
        
        # Initialize engine (user=None for now, can add auth later)
        engine = PaperTradingEngine(initial_balance=10000.0, user=None)
        
        # Execute trade
        trade = engine.execute_order(
            pair=pair,
            order_type=order_type,
            entry_price=current_price,
            stop_loss=float(stop_loss) if stop_loss else (current_price * 0.98 if signal == 'bullish' else current_price * 1.02),
            take_profit_1=take_profit,
            lot_size=lot_size,
            signal_source='ml_signal',
            notes=f"Auto-executed from signal (confidence: {float(probability)*100:.1f}%)"
        )
        
        logger.info(f"Paper trade executed: {pair} {order_type} @ {current_price}")
        
        return Response({
            'success': True,
            'trade_id': trade.id,
            'pair': trade.pair,
            'order_type': order_type,
            'entry_price': float(trade.entry_price),
            'stop_loss': float(trade.stop_loss),
            'take_profit': float(trade.take_profit_1) if trade.take_profit_1 else None,
            'lot_size': float(trade.lot_size),
            'timestamp': trade.entry_time.isoformat(),
            'message': f'Paper trade executed successfully for {pair}'
        })
        
    except Exception as e:
        logger.error(f"Error executing paper trade: {e}")
        return Response({
            'success': False,
            'error': str(e)
        }, status=500)


# Grade thresholds for signal accuracy
SIGNAL_GRADES = [
    (0.75, 'A++', '#00e5ff'),
    (0.70, 'A+',  '#00ff87'),
    (0.65, 'A',   '#60efff'),
    (0.60, 'A-',  '#b8ff45'),
    (0.55, 'B+',  '#ffd700'),
    (0.50, 'B',   '#ffa500'),
    (0.00, 'C',   '#ff6b6b'),
]

def _grade(accuracy):
    for threshold, grade, color in SIGNAL_GRADES:
        if accuracy >= threshold:
            return grade, color
    return 'C', '#ff6b6b'


# Human-readable labels for well-known signal names
SIGNAL_LABELS = {
    'smc_signal': 'SMC Signal',
    'order_block_support': 'Order Block Support',
    'master_signal': 'Master Signal',
    'master_signal_raw': 'Master Signal (Raw)',
    'rsi_mean_reversion': 'RSI Mean Reversion',
    'elliott_wave_signal': 'Elliott Wave',
    'holloway_bear_signal': 'Holloway Bear',
    'double_bottom': 'Double Bottom',
    'signal_confluence_count': 'Signal Confluence Count',
    'position_size_factor': 'Position Size Factor',
}

def _label(feature):
    if feature in SIGNAL_LABELS:
        return SIGNAL_LABELS[feature]
    # Convert snake_case to Title Case as fallback
    return feature.replace('_', ' ').title()


@api_view(['GET'])
@permission_classes([AllowAny])
def signal_performance(request):
    """
    Return per-signal accuracy data from the pre-computed evaluation CSVs
    (EURUSD_signal_evaluation.csv / XAUUSD_signal_evaluation.csv).

    Query params:
        pair:     EURUSD | XAUUSD | all  (default: all)
        min_acc:  minimum accuracy to include, 0-1  (default: 0.50)
        limit:    max signals per pair  (default: 20)
    """
    pair_param = request.GET.get('pair', 'all').upper()
    min_acc = float(request.GET.get('min_acc', 0.50))
    limit = int(request.GET.get('limit', 20))

    pairs_to_load = ['EURUSD', 'XAUUSD'] if pair_param == 'ALL' else [pair_param]

    # Resolve CSV directory: try project root first, then cwd
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root

    result = {}
    for p in pairs_to_load:
        filename = f'{p}_signal_evaluation.csv'
        # Search in project root, then cwd
        csv_path = None
        for candidate in [os.path.join(base_dir, filename), os.path.join(os.getcwd(), filename)]:
            if os.path.exists(candidate):
                csv_path = candidate
                break
        if csv_path is None:
            result[p] = {'error': f'Evaluation file not found: {filename}', 'signals': []}
            continue

        try:
            df = pd.read_csv(csv_path)
            df = df[df['accuracy'] >= min_acc].sort_values('accuracy', ascending=False).head(limit)

            signals_out = []
            for _, row in df.iterrows():
                acc = float(row['accuracy'])
                grade, grade_color = _grade(acc)
                signals_out.append({
                    'feature': row['feature'],
                    'label': _label(row['feature']),
                    'accuracy': round(acc, 4),
                    'accuracy_pct': round(acc * 100, 2),
                    'hit_rate': round(float(row['hit_rate']), 4),
                    'correlation': round(float(row['correlation']), 4),
                    'grade': grade,
                    'grade_color': grade_color,
                })
            result[p] = {'signals': signals_out, 'count': len(signals_out)}
        except Exception as e:
            logger.error(f"Error reading {csv_path}: {e}")
            result[p] = {'error': str(e), 'signals': []}

    return Response(result)
