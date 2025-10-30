"""
Paper Trading REST API Views
Django REST Framework views for paper trading operations
"""
import logging
from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from django.utils import timezone
from datetime import timedelta

from .models import PaperTrade, PerformanceMetrics, PriceCache
from .serializers import (
    PaperTradeSerializer,
    PerformanceMetricsSerializer,
    PriceCacheSerializer,
    TradeExecutionSerializer
)
from .engine import PaperTradingEngine
from .data_aggregator import DataAggregator
from .mt_bridge import MT5EasyBridge

logger = logging.getLogger(__name__)


class PaperTradeViewSet(viewsets.ModelViewSet):
    """
    API endpoint for paper trades
    """
    queryset = PaperTrade.objects.all().order_by('-entry_time')
    serializer_class = PaperTradeSerializer
    permission_classes = [IsAuthenticated]
    
    def create(self, request, *args, **kwargs):
        """Disable direct trade creation - use execute action instead"""
        return Response(
            {'error': 'Direct trade creation not allowed. Use /execute/ endpoint.'},
            status=status.HTTP_405_METHOD_NOT_ALLOWED
        )
    
    def get_queryset(self):
        """Filter trades by query parameters and user"""
        queryset = super().get_queryset()
        
        # Filter by current user
        if self.request.user.is_authenticated:
            queryset = queryset.filter(user=self.request.user)
        
        pair = self.request.query_params.get('pair', None)
        status_filter = self.request.query_params.get('status', None)
        signal_type = self.request.query_params.get('signal_type', None)
        days = self.request.query_params.get('days', None)
        
        if pair:
            queryset = queryset.filter(pair=pair)
        
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        if signal_type:
            queryset = queryset.filter(signal_type=signal_type)
        
        if days:
            cutoff = timezone.now() - timedelta(days=int(days))
            queryset = queryset.filter(entry_time__gte=cutoff)
        
        return queryset
    
    @action(detail=False, methods=['post'])
    def execute(self, request):
        """
        Execute a new paper trade
        POST /api/paper-trades/execute/
        """
        serializer = TradeExecutionSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST
            )
        
        data = serializer.validated_data
        
        try:
            engine = PaperTradingEngine(user=request.user)
            trade = engine.execute_order(
                pair=data['pair'],
                order_type=data['order_type'],
                entry_price=data['entry_price'],
                stop_loss=data['stop_loss'],
                take_profit_1=data.get('take_profit_1'),
                take_profit_2=data.get('take_profit_2'),
                take_profit_3=data.get('take_profit_3'),
                lot_size=data.get('lot_size', 0.01),
                signal_id=data.get('signal_id'),
                signal_type=data.get('signal_type'),
                signal_source=data.get('signal_source'),
                notes=data.get('notes')
            )
            
            return Response(
                PaperTradeSerializer(trade).data,
                status=status.HTTP_201_CREATED
            )
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'])
    def close(self, request, pk=None):
        """
        Close an open trade
        POST /api/paper-trades/{id}/close/
        """
        trade = self.get_object()
        
        if trade.status != 'open':
            return Response(
                {'error': 'Trade is not open'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        exit_price = request.data.get('exit_price')
        
        if not exit_price:
            return Response(
                {'error': 'exit_price is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            engine = PaperTradingEngine(user=request.user)
            engine.close_position(trade.id, float(exit_price), 'manual')
            
            trade.refresh_from_db()
            
            return Response(
                PaperTradeSerializer(trade).data,
                status=status.HTTP_200_OK
            )
            
        except Exception as e:
            logger.error(f"Trade close error: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'])
    def open_positions(self, request):
        """
        Get all open positions
        GET /api/paper-trades/open_positions/
        """
        engine = PaperTradingEngine(user=request.user)
        positions = engine.get_open_positions()
        
        return Response(
            PaperTradeSerializer(positions, many=True).data,
            status=status.HTTP_200_OK
        )
    
    @action(detail=False, methods=['get'])
    def performance(self, request):
        """
        Get performance summary
        GET /api/paper-trades/performance/?days=30
        """
        days = int(request.query_params.get('days', 30))
        
        engine = PaperTradingEngine(user=request.user)
        summary = engine.get_performance_summary(days=days)
        
        return Response(summary, status=status.HTTP_200_OK)
    
    @action(detail=False, methods=['get'])
    def equity_curve(self, request):
        """
        Get equity curve data
        GET /api/paper-trades/equity_curve/?days=30
        """
        days = int(request.query_params.get('days', 30))
        
        engine = PaperTradingEngine(user=request.user)
        curve = engine.get_equity_curve(days=days)
        
        return Response(curve, status=status.HTTP_200_OK)
    
    @action(detail=False, methods=['post'])
    def update_positions(self, request):
        """
        Update all open positions with current prices
        POST /api/paper-trades/update_positions/
        """
        prices = request.data.get('prices', {})
        
        if not prices:
            # If no prices provided, return success with no updates
            return Response({
                'success': True,
                'updated': 0,
                'closed_trades': [],
                'message': 'No prices provided'
            }, status=status.HTTP_200_OK)
        
        try:
            engine = PaperTradingEngine(user=request.user)
            closed_trades = engine.update_positions(prices)
            
            return Response({
                'success': True,
                'updated': len(closed_trades),
                'closed_trades': closed_trades,
                'message': f'Updated positions, closed {len(closed_trades)} trades'
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Position update error: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@api_view(['GET'])
@permission_classes([AllowAny])
def get_realtime_price(request):
    """
    Get real-time price for a symbol
    GET /api/price/realtime/?symbol=EURUSD
    """
    symbol = request.query_params.get('symbol')
    
    if not symbol:
        return Response(
            {'error': 'symbol parameter is required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        aggregator = DataAggregator()
        price = aggregator.get_realtime_price(symbol)
        
        if not price:
            return Response(
                {'error': f'Could not get price for {symbol}'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        return Response(price, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Price fetch error: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([AllowAny])
def get_historical_ohlc(request):
    """
    Get historical OHLC data
    GET /api/price/ohlc/?symbol=EURUSD&interval=1h&limit=100
    """
    symbol = request.query_params.get('symbol')
    interval = request.query_params.get('interval', '1h')
    limit = int(request.query_params.get('limit', 100))
    
    if not symbol:
        return Response(
            {'error': 'symbol parameter is required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        aggregator = DataAggregator()
        df = aggregator.get_historical_ohlc(symbol, interval, limit)
        
        if df is None or df.empty:
            return Response(
                {'error': f'Could not get historical data for {symbol}'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Convert DataFrame to list of dicts
        data = df.to_dict('records')
        
        # Convert timestamps to strings
        for item in data:
            if 'timestamp' in item:
                item['timestamp'] = item['timestamp'].isoformat()
        
        return Response({
            'symbol': symbol,
            'interval': interval,
            'data': data
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Historical data error: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([AllowAny])
def update_positions(request):
    """
    Update all open positions with current prices
    Checks for SL/TP hits
    POST /api/positions/update/
    Body: {"prices": {"EURUSD": 1.0850, "XAUUSD": 2650.00}}
    """
    prices = request.data.get('prices', {})
    
    if not prices:
        return Response(
            {'error': 'prices dict is required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        engine = PaperTradingEngine()
        closed_trades = engine.update_positions(prices)
        
        return Response({
            'success': True,
            'closed_trades': closed_trades,
            'message': f'Updated positions, closed {len(closed_trades)} trades'
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Position update error: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([AllowAny])
def mt_account_info(request):
    """
    Get MetaTrader account info (simulated for paper trading)
    GET /api/mt/account/
    """
    try:
        bridge = MT5EasyBridge()
        account = bridge.get_account_info()
        
        return Response(account, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"MT account error: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([AllowAny])
def mt_positions(request):
    """
    Get MetaTrader positions (from paper trading database)
    GET /api/mt/positions/
    """
    try:
        bridge = MT5EasyBridge()
        positions = bridge.get_positions()
        
        return Response(positions, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"MT positions error: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class PerformanceMetricsViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API endpoint for performance metrics
    """
    queryset = PerformanceMetrics.objects.all().order_by('-date')
    serializer_class = PerformanceMetricsSerializer
    permission_classes = [AllowAny]
    
    def get_queryset(self):
        """Filter metrics by query parameters"""
        queryset = super().get_queryset()
        
        pair = self.request.query_params.get('pair', None)
        days = self.request.query_params.get('days', None)
        
        if pair:
            queryset = queryset.filter(pair=pair)
        
        if days:
            cutoff = timezone.now().date() - timedelta(days=int(days))
            queryset = queryset.filter(date__gte=cutoff)
        
        return queryset
