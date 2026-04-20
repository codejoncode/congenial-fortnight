from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.management import call_command
import json
from io import StringIO
import traceback
from pathlib import Path
from datetime import datetime

@csrf_exempt
def generate_signal(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    try:
        body = json.loads(request.body) if request.body else {}
        pair = body.get('pair', 'all')
        output = StringIO()
        call_command('generate_daily_signal', '--pair', pair, stdout=output)
        date_str = datetime.now().strftime('%Y%m%d')
        signals_file = Path('signals') / f'signals_{date_str}.json'
        if signals_file.exists():
            with open(signals_file, 'r') as f:
                signals = json.load(f)
        else:
            signals = []
        return JsonResponse({'success': True, 'signals': signals, 'message': 'Signals generated successfully'})
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@csrf_exempt
def update_data(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    try:
        body = json.loads(request.body) if request.body else {}
        pairs = body.get('pairs', 'all')
        output = StringIO()
        if pairs == 'all':
            call_command('update_data', '--all', stdout=output)
        else:
            call_command('update_data', '--pair', pairs, stdout=output)
        return JsonResponse({'success': True, 'message': 'Data updated successfully', 'details': output.getvalue()})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@csrf_exempt
def run_backtest(request):
    pair = request.GET.get('pair', 'EURUSD')
    days = request.GET.get('days', '30')
    try:
        from daily_forex_signal_system import DailyForexSignal
        signal_sys = DailyForexSignal()
        results = signal_sys.backtest_last_n_days_enhanced(pair, n=int(days))
        return JsonResponse(results)
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': f'Backtest failed: {str(e)}', 'error': str(e)}, status=500)
