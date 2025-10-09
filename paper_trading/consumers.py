"""
WebSocket Consumer for Real-Time Trading Updates
Uses Django Channels for WebSocket support
"""
import json
import logging
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from datetime import datetime

logger = logging.getLogger(__name__)


class TradingWebSocketConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time trading updates
    Broadcasts price updates, signal alerts, and trade executions
    """
    
    async def connect(self):
        """Handle WebSocket connection"""
        self.room_group_name = 'trading_updates'
        
        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        
        await self.accept()
        logger.info(f"✅ WebSocket connected: {self.channel_name}")
        
        # Send initial connection confirmation
        await self.send(text_data=json.dumps({
            'type': 'connection',
            'message': 'Connected to trading updates',
            'timestamp': datetime.now().isoformat()
        }))
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
        logger.info(f"❌ WebSocket disconnected: {self.channel_name}")
    
    async def receive(self, text_data):
        """Receive message from WebSocket"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            
            if message_type == 'subscribe':
                # Subscribe to specific symbols
                symbols = data.get('symbols', [])
                await self.handle_subscribe(symbols)
            
            elif message_type == 'unsubscribe':
                # Unsubscribe from symbols
                symbols = data.get('symbols', [])
                await self.handle_unsubscribe(symbols)
            
            elif message_type == 'ping':
                # Respond to ping
                await self.send(text_data=json.dumps({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                }))
        
        except json.JSONDecodeError:
            logger.error("Invalid JSON received")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Invalid JSON format'
            }))
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    async def handle_subscribe(self, symbols):
        """Handle symbol subscription"""
        await self.send(text_data=json.dumps({
            'type': 'subscribed',
            'symbols': symbols,
            'timestamp': datetime.now().isoformat()
        }))
    
    async def handle_unsubscribe(self, symbols):
        """Handle symbol unsubscription"""
        await self.send(text_data=json.dumps({
            'type': 'unsubscribed',
            'symbols': symbols,
            'timestamp': datetime.now().isoformat()
        }))
    
    # Receive messages from room group
    
    async def price_update(self, event):
        """Send price update to WebSocket"""
        await self.send(text_data=json.dumps({
            'type': 'price_update',
            'symbol': event['symbol'],
            'data': event['data'],
            'timestamp': event['timestamp']
        }))
    
    async def signal_alert(self, event):
        """Send signal alert to WebSocket"""
        await self.send(text_data=json.dumps({
            'type': 'signal_alert',
            'signal': event['signal'],
            'timestamp': event['timestamp']
        }))
    
    async def trade_execution(self, event):
        """Send trade execution notification to WebSocket"""
        await self.send(text_data=json.dumps({
            'type': 'trade_execution',
            'trade': event['trade'],
            'timestamp': event['timestamp']
        }))
    
    async def trade_closed(self, event):
        """Send trade closed notification to WebSocket"""
        await self.send(text_data=json.dumps({
            'type': 'trade_closed',
            'trade': event['trade'],
            'reason': event.get('reason', 'manual'),
            'timestamp': event['timestamp']
        }))


class PriceStreamConsumer(AsyncWebsocketConsumer):
    """
    Dedicated consumer for high-frequency price streams
    Optimized for minimal latency
    """
    
    async def connect(self):
        """Handle connection"""
        self.symbols = []
        await self.accept()
        logger.info(f"📡 Price stream connected: {self.channel_name}")
    
    async def disconnect(self, close_code):
        """Handle disconnection"""
        logger.info(f"📡 Price stream disconnected: {self.channel_name}")
    
    async def receive(self, text_data):
        """Handle incoming messages"""
        try:
            data = json.loads(text_data)
            action = data.get('action')
            
            if action == 'subscribe':
                symbols = data.get('symbols', [])
                self.symbols.extend(symbols)
                
                # Start price streaming for these symbols
                asyncio.create_task(self.stream_prices(symbols))
                
                await self.send(text_data=json.dumps({
                    'action': 'subscribed',
                    'symbols': symbols
                }))
            
            elif action == 'unsubscribe':
                symbols = data.get('symbols', [])
                self.symbols = [s for s in self.symbols if s not in symbols]
                
                await self.send(text_data=json.dumps({
                    'action': 'unsubscribed',
                    'symbols': symbols
                }))
        
        except Exception as e:
            logger.error(f"Price stream error: {e}")
    
    async def stream_prices(self, symbols):
        """Stream prices for subscribed symbols"""
        from .data_aggregator import DataAggregator
        
        aggregator = DataAggregator()
        
        while True:
            try:
                for symbol in symbols:
                    if symbol not in self.symbols:
                        # Symbol unsubscribed
                        continue
                    
                    # Get price
                    price = await database_sync_to_async(
                        aggregator.get_realtime_price
                    )(symbol)
                    
                    if price:
                        await self.send(text_data=json.dumps({
                            'type': 'price',
                            'symbol': symbol,
                            'bid': price['bid'],
                            'ask': price['ask'],
                            'time': price['time']
                        }))
                
                # Wait before next update (1-5 seconds based on free tier limits)
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Price streaming error: {e}")
                await asyncio.sleep(10)
