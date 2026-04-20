from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.management import call_command
import json
from pathlib import Path

@api_view(['POST'])
def update_data_api(request):
    pairs = request.data.get('pairs', 'all')
    if pairs == 'all':
        call_command('update_data', '--all')
    else:
        call_command('update_data', '--pair', pairs)
    return Response({'success': True, 'message': 'Data updated successfully'})

@api_view(['POST'])
def generate_signal_api(request):
    pair = request.data.get('pair', 'all')
    if pair == 'all':
        call_command('generate_daily_signal', '--all')
    else:
        call_command('generate_daily_signal', '--pair', pair)
    # Load latest signals JSON
    from datetime import datetime
    today = datetime.now().strftime('%Y%m%d')
    out_path = Path(f'signals/signals_{today}.json')
    if out_path.exists():
        with open(out_path) as f:
            signals = json.load(f)
    else:
        signals = []
    return Response({'success': True, 'signals': signals, 'message': 'Signal generation complete'})
