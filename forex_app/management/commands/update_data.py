import pandas as pd
import requests
from django.core.management.base import BaseCommand
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Incrementally update forex data for EURUSD and XAUUSD'

    def add_arguments(self, parser):
        parser.add_argument('--pair', type=str, default=None, help='Currency pair to update')
        parser.add_argument('--all', action='store_true', help='Update all pairs')

    def handle(self, *args, **options):
        pairs = ['EURUSD', 'XAUUSD'] if options['all'] or not options['pair'] else [options['pair']]
        for pair in pairs:
            self.update_pair(pair)

    def update_pair(self, pair):
        # Historical CSV path
        csv_path = Path(f'data/{pair}_historical.csv')
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Load existing data
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            last_date = pd.to_datetime(df['timestamp']).max()
        else:
            df = pd.DataFrame()
            last_date = None
        # Fetch new data (stub: replace with real API)
        new_data = self.fetch_new_data(pair, last_date)
        if new_data.empty:
            logger.info(f'No new data for {pair}')
            return
        # Append only new rows
        combined = pd.concat([df, new_data]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        combined.to_csv(csv_path, index=False)
        logger.info(f'Updated {pair}: {len(new_data)} new rows, last date {combined["timestamp"].max()}')
        self.stdout.write(self.style.SUCCESS(f'Updated {pair}: {len(new_data)} new rows, last date {combined["timestamp"].max()}'))

    def fetch_new_data(self, pair, last_date):
        # Replace with real API logic
        # For now, just return empty DataFrame
        return pd.DataFrame()