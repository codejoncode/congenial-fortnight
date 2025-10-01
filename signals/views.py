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
        # Load historical data
        import os
        data_path = f'data/raw/{pair}_Daily.csv'
        
        if not os.path.exists(data_path):
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
