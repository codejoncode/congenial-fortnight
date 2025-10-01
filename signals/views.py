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
    
    # Load data
    data_path = 'data/raw'
    file_path = f'{data_path}/{pair}_Daily.csv'
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Simple backtest simulation (simplified version)
        if len(df) < days + 20:
            return Response({'error': 'Not enough data for backtest'})
        
        # Get last n days
        recent_df = df.tail(days + 20)
        
        # Simulate signals (this is simplified - in reality use the full model)
        results = []
        for i in range(days):
            if i + 1 < len(recent_df):
                current = recent_df.iloc[-(i+2)]  # Day before
                next_day = recent_df.iloc[-(i+1)]  # Next day
                actual_move = 'bullish' if next_day['close'] > current['close'] else 'bearish'
                
                # Mock signal based on some logic (replace with real model)
                mock_prob = np.random.uniform(0.1, 0.9)
                mock_signal = 'bullish' if mock_prob > 0.5 else 'bearish'
                correct = (mock_signal == actual_move)
                
                results.append({
                    'date': current.name.strftime('%Y-%m-%d'),
                    'signal': mock_signal,
                    'actual': actual_move,
                    'correct': correct,
                    'probability': mock_prob
                })
        
        # Calculate stats
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        accuracy = correct / total if total > 0 else 0
        
        # By probability bins
        prob_bins = {
            '0.0-0.2': {'total': 0, 'correct': 0},
            '0.2-0.4': {'total': 0, 'correct': 0},
            '0.4-0.6': {'total': 0, 'correct': 0},
            '0.6-0.8': {'total': 0, 'correct': 0},
            '0.8-1.0': {'total': 0, 'correct': 0}
        }
        
        for r in results:
            prob = r['probability']
            if prob < 0.2:
                bin_key = '0.0-0.2'
            elif prob < 0.4:
                bin_key = '0.2-0.4'
            elif prob < 0.6:
                bin_key = '0.4-0.6'
            elif prob < 0.8:
                bin_key = '0.6-0.8'
            else:
                bin_key = '0.8-1.0'
            
            prob_bins[bin_key]['total'] += 1
            if r['correct']:
                prob_bins[bin_key]['correct'] += 1
        
        for bin_key in prob_bins:
            total_bin = prob_bins[bin_key]['total']
            prob_bins[bin_key]['accuracy'] = prob_bins[bin_key]['correct'] / total_bin if total_bin > 0 else 0
        
        return Response({
            'pair': pair,
            'days': days,
            'overall_accuracy': accuracy,
            'total_signals': total,
            'probability_bins': prob_bins,
            'recent_results': results[:10]  # Last 10 for display
        })
        
    except Exception as e:
        return Response({'error': str(e)})
