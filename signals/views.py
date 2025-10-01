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
        sys.path.append(os.getcwd())
        from daily_forex_signal_system import DailyForexSignal

        # Run enhanced backtest
        ds = DailyForexSignal()
        result = ds.backtest_last_n_days_enhanced(pair, n=days)

        # Format response for frontend
        response_data = {
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
        return Response({'error': str(e)})
