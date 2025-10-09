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
        model_file = f'models/{pair}_pip_based_model.joblib'
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
def get_signals(request, pair):
    """Get trading signals for a specific currency pair"""
    try:
        # Initialize trading system
        data_collector = TradingDataCollector()
        strategies = TradingStrategies(data_collector)

        # Load data for the pair
        data = data_collector.collect_all_data()

        if pair not in data or data[pair].empty:
            return Response({
                'error': f'No data available for {pair}',
                'available_pairs': list(data.keys())
            }, status=404)

        df = data[pair]

        # Generate signals using master signal system
        signals = strategies.master_signal_system(df)

        # Get latest signal
        latest_signal = signals.iloc[-1] if not signals.empty else None

        # Get recent signals (last 10)
        recent_signals = []
        for i in range(max(0, len(signals)-10), len(signals)):
            recent_signals.append({
                'date': signals.index[i].strftime('%Y-%m-%d'),
                'signal': 'bullish' if signals.iloc[i] > 0 else 'bearish',
                'strength': abs(signals.iloc[i])
            })

        return Response({
            'pair': pair,
            'latest_signal': {
                'date': signals.index[-1].strftime('%Y-%m-%d') if latest_signal is not None else None,
                'signal': 'bullish' if latest_signal and latest_signal > 0 else 'bearish',
                'strength': abs(latest_signal) if latest_signal else 0
            },
            'recent_signals': recent_signals,
            'data_points': len(df)
        })

    except Exception as e:
        logger.error(f"Error getting signals for {pair}: {str(e)}")
        return Response({
            'error': f'Failed to get signals for {pair}: {str(e)}'
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
def update_data(request):
    """Update trading data from APIs"""
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
