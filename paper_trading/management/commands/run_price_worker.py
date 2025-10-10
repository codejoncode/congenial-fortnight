"""
Management command to run price update worker
Updates positions and checks for SL/TP hits
"""
import logging
import time
from django.core.management.base import BaseCommand
from django.utils import timezone
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

from paper_trading.engine import PaperTradingEngine
from paper_trading.data_aggregator import DataAggregator

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Run price update worker for paper trading'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--interval',
            type=int,
            default=5,
            help='Update interval in seconds (default: 5)'
        )
        parser.add_argument(
            '--pairs',
            type=str,
            default='EURUSD,XAUUSD',
            help='Comma-separated list of pairs to monitor'
        )
    
    def handle(self, *args, **options):
        interval = options['interval']
        pairs = options['pairs'].split(',')
        
        self.stdout.write(
            self.style.SUCCESS(
                f'🚀 Starting price update worker\n'
                f'   Interval: {interval}s\n'
                f'   Pairs: {", ".join(pairs)}'
            )
        )
        
        engine = PaperTradingEngine()
        aggregator = DataAggregator()
        channel_layer = get_channel_layer()
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                self.stdout.write(f'\n📊 Update #{iteration} at {timezone.now()}')
                
                # Get current prices
                prices = {}
                for pair in pairs:
                    price_data = aggregator.get_realtime_price(pair)
                    
                    if price_data:
                        mid_price = (price_data['bid'] + price_data['ask']) / 2
                        prices[pair] = mid_price
                        
                        self.stdout.write(
                            f'   {pair}: {mid_price:.5f} (bid: {price_data["bid"]:.5f}, ask: {price_data["ask"]:.5f})'
                        )
                        
                        # Broadcast price update via WebSocket
                        if channel_layer:
                            async_to_sync(channel_layer.group_send)(
                                'trading_updates',
                                {
                                    'type': 'price_update',
                                    'symbol': pair,
                                    'data': price_data,
                                    'timestamp': timezone.now().isoformat()
                                }
                            )
                    else:
                        self.stdout.write(
                            self.style.WARNING(f'   ⚠️ Failed to get price for {pair}')
                        )
                
                # Update open positions
                if prices:
                    closed_trades = engine.update_positions(prices)
                    
                    if closed_trades:
                        self.stdout.write(
                            self.style.SUCCESS(
                                f'\n✅ Closed {len(closed_trades)} trades:'
                            )
                        )
                        
                        for closed in closed_trades:
                            self.stdout.write(
                                f'   {closed["pair"]} - {closed["exit_reason"]} @ {closed["exit_price"]:.5f} '
                                f'({closed["pips"]:+.1f} pips, ${closed["pnl"]:+.2f})'
                            )
                            
                            # Broadcast trade closed via WebSocket
                            if channel_layer:
                                async_to_sync(channel_layer.group_send)(
                                    'trading_updates',
                                    {
                                        'type': 'trade_closed',
                                        'trade': closed,
                                        'reason': closed['exit_reason'],
                                        'timestamp': timezone.now().isoformat()
                                    }
                                )
                
                # Sleep until next update
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.stdout.write(
                self.style.WARNING('\n\n⏹️ Price update worker stopped by user')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'\n\n❌ Error in price update worker: {e}')
            )
            logger.exception('Price update worker error')
            raise
