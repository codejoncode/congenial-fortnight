"""
MetaTrader Bridge
Connects to MetaTrader 4/5 for data and order simulation
Supports both MetaTrader5 Python package and ZeroMQ bridge
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class MetaTraderBridge:
    """
    Bridge to MetaTrader 4/5 platform
    Provides price data and simulates order execution
    """
    
    def __init__(self, mode: str = 'mt5_python'):
        """
        Initialize MetaTrader bridge
        
        Args:
            mode: 'mt5_python' (MT5 Python package) or 'zeromq' (ZeroMQ bridge)
        """
        self.mode = mode
        self.connected = False
        
        if mode == 'mt5_python':
            self.mt5 = self._init_mt5_python()
        elif mode == 'zeromq':
            self.zmq_socket = self._init_zeromq()
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'mt5_python' or 'zeromq'")
    
    def _init_mt5_python(self):
        """Initialize MT5 Python package"""
        try:
            import MetaTrader5 as mt5
            
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return None
            
            logger.info("✅ MT5 Python package initialized")
            self.connected = True
            return mt5
            
        except ImportError:
            logger.warning("⚠️ MetaTrader5 package not installed. Install with: pip install MetaTrader5")
            return None
        except Exception as e:
            logger.error(f"MT5 init error: {e}")
            return None
    
    def _init_zeromq(self):
        """Initialize ZeroMQ connection"""
        try:
            import zmq
            
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.connect("tcp://localhost:5555")
            
            logger.info("✅ ZeroMQ bridge connected")
            self.connected = True
            return socket
            
        except ImportError:
            logger.warning("⚠️ pyzmq not installed. Install with: pip install pyzmq")
            return None
        except Exception as e:
            logger.error(f"ZeroMQ init error: {e}")
            return None
    
    def connect(self) -> bool:
        """Check/establish connection"""
        if self.mode == 'mt5_python' and self.mt5:
            return True
        elif self.mode == 'zeromq' and self.zmq_socket:
            return True
        return False
    
    def disconnect(self):
        """Disconnect from MetaTrader"""
        if self.mode == 'mt5_python' and self.mt5:
            self.mt5.shutdown()
            logger.info("MT5 disconnected")
        elif self.mode == 'zeromq' and self.zmq_socket:
            self.zmq_socket.close()
            logger.info("ZeroMQ disconnected")
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """
        Get current bid/ask price
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            
        Returns:
            Dict with {bid, ask, time} or None
        """
        if self.mode == 'mt5_python':
            return self._get_price_mt5(symbol)
        elif self.mode == 'zeromq':
            return self._get_price_zmq(symbol)
        return None
    
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str = 'H1',
        bars: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Get historical OHLC data
        
        Args:
            symbol: Trading symbol
            timeframe: MT timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
            bars: Number of bars to retrieve
            
        Returns:
            DataFrame with OHLC data or None
        """
        if self.mode == 'mt5_python':
            return self._get_bars_mt5(symbol, timeframe, bars)
        elif self.mode == 'zeromq':
            return self._get_bars_zmq(symbol, timeframe, bars)
        return None
    
    def simulate_order(
        self,
        order_type: str,
        symbol: str,
        volume: float,
        sl: float = None,
        tp: float = None
    ) -> Dict:
        """
        Simulate order execution (paper trading)
        Does NOT place real orders
        
        Args:
            order_type: 'buy' or 'sell'
            symbol: Trading symbol
            volume: Lot size
            sl: Stop loss price
            tp: Take profit price
            
        Returns:
            Dict with order details
        """
        price = self.get_current_price(symbol)
        
        if not price:
            return {
                'success': False,
                'error': 'Could not get current price'
            }
        
        entry_price = price['ask'] if order_type == 'buy' else price['bid']
        
        return {
            'success': True,
            'order_id': self._generate_order_id(),
            'symbol': symbol,
            'type': order_type,
            'volume': volume,
            'entry_price': entry_price,
            'sl': sl,
            'tp': tp,
            'time': datetime.now().isoformat(),
            'status': 'filled'
        }
    
    def get_account_info(self) -> Optional[Dict]:
        """Get account information (for paper trading, returns simulated data)"""
        return {
            'balance': 10000.0,
            'equity': 10000.0,
            'margin': 0.0,
            'free_margin': 10000.0,
            'leverage': 100,
            'currency': 'USD'
        }
    
    def get_positions(self) -> List[Dict]:
        """Get open positions (for paper trading, returns from database)"""
        from .models import PaperTrade
        
        open_trades = PaperTrade.objects.filter(status='open')
        
        positions = []
        for trade in open_trades:
            positions.append({
                'ticket': trade.id,
                'symbol': trade.pair,
                'type': trade.order_type,
                'volume': float(trade.lot_size),
                'price_open': float(trade.entry_price),
                'sl': float(trade.stop_loss),
                'tp': float(trade.take_profit_1) if trade.take_profit_1 else None,
                'time': trade.entry_time.isoformat(),
                'profit': float(trade.profit_loss) if trade.profit_loss else 0.0
            })
        
        return positions
    
    # Private methods for MT5 Python
    
    def _get_price_mt5(self, symbol: str) -> Optional[Dict]:
        """Get price using MT5 Python"""
        if not self.mt5:
            return None
        
        try:
            tick = self.mt5.symbol_info_tick(symbol)
            
            if not tick:
                logger.warning(f"No tick data for {symbol}")
                return None
            
            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'time': datetime.fromtimestamp(tick.time).isoformat()
            }
        except Exception as e:
            logger.error(f"MT5 get_price error: {e}")
            return None
    
    def _get_bars_mt5(self, symbol: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        """Get OHLC bars using MT5 Python"""
        if not self.mt5:
            return None
        
        try:
            # Map timeframe string to MT5 constant
            tf_map = {
                'M1': self.mt5.TIMEFRAME_M1,
                'M5': self.mt5.TIMEFRAME_M5,
                'M15': self.mt5.TIMEFRAME_M15,
                'M30': self.mt5.TIMEFRAME_M30,
                'H1': self.mt5.TIMEFRAME_H1,
                'H4': self.mt5.TIMEFRAME_H4,
                'D1': self.mt5.TIMEFRAME_D1,
                'W1': self.mt5.TIMEFRAME_W1,
                'MN1': self.mt5.TIMEFRAME_MN1,
            }
            
            mt5_timeframe = tf_map.get(timeframe, self.mt5.TIMEFRAME_H1)
            
            rates = self.mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No rates for {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={'time': 'timestamp', 'tick_volume': 'volume'})
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"MT5 get_bars error: {e}")
            return None
    
    # Private methods for ZeroMQ
    
    def _get_price_zmq(self, symbol: str) -> Optional[Dict]:
        """Get price using ZeroMQ"""
        if not self.zmq_socket:
            return None
        
        try:
            import json
            
            command = {
                'action': 'GET_PRICE',
                'symbol': symbol
            }
            
            self.zmq_socket.send_json(command)
            response = self.zmq_socket.recv_json(flags=0)
            
            if response.get('success'):
                return response['data']
            else:
                logger.error(f"ZMQ error: {response.get('error')}")
                return None
                
        except Exception as e:
            logger.error(f"ZeroMQ get_price error: {e}")
            return None
    
    def _get_bars_zmq(self, symbol: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        """Get OHLC bars using ZeroMQ"""
        if not self.zmq_socket:
            return None
        
        try:
            import json
            
            command = {
                'action': 'GET_BARS',
                'symbol': symbol,
                'timeframe': timeframe,
                'count': bars
            }
            
            self.zmq_socket.send_json(command)
            response = self.zmq_socket.recv_json(flags=0)
            
            if response.get('success'):
                return pd.DataFrame(response['data'])
            else:
                logger.error(f"ZMQ error: {response.get('error')}")
                return None
                
        except Exception as e:
            logger.error(f"ZeroMQ get_bars error: {e}")
            return None
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        import uuid
        return f"PT{uuid.uuid4().hex[:12].upper()}"


class MT5EasyBridge(MetaTraderBridge):
    """Simplified MT5 bridge with automatic fallback to data aggregator"""
    
    def __init__(self, data_aggregator=None):
        """
        Initialize with optional data aggregator fallback
        
        Args:
            data_aggregator: DataAggregator instance for fallback
        """
        try:
            super().__init__(mode='mt5_python')
        except:
            logger.warning("⚠️ MT5 not available, using data aggregator fallback")
            self.mt5 = None
            self.connected = False
        
        self.data_aggregator = data_aggregator
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get price from MT5 or fallback to data aggregator"""
        if self.connected and self.mt5:
            price = super().get_current_price(symbol)
            if price:
                return price
        
        # Fallback to data aggregator
        if self.data_aggregator:
            logger.debug(f"Using data aggregator fallback for {symbol}")
            return self.data_aggregator.get_realtime_price(symbol)
        
        return None
    
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str = 'H1',
        bars: int = 100
    ) -> Optional[pd.DataFrame]:
        """Get historical data from MT5 or fallback to data aggregator"""
        if self.connected and self.mt5:
            df = super().get_historical_data(symbol, timeframe, bars)
            if df is not None and not df.empty:
                return df
        
        # Fallback to data aggregator
        if self.data_aggregator:
            logger.debug(f"Using data aggregator fallback for {symbol} history")
            # Map MT timeframe to standard format
            tf_map = {
                'M1': '1m', 'M5': '5m', 'M15': '15m', 'M30': '30m',
                'H1': '1h', 'H4': '4h', 'D1': '1d', 'W1': '1w'
            }
            interval = tf_map.get(timeframe, '1h')
            return self.data_aggregator.get_historical_ohlc(symbol, interval, bars)
        
        return None
