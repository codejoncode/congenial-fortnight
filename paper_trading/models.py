"""
Paper Trading Database Models
Django models for tracking simulated trades, positions, and performance
"""
from django.db import models
from django.utils import timezone
from decimal import Decimal


class PaperTrade(models.Model):
    """Represents a single paper trade (simulated order)"""
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('open', 'Open'),
        ('closed', 'Closed'),
        ('cancelled', 'Cancelled'),
    ]
    
    ORDER_TYPE_CHOICES = [
        ('buy', 'Buy'),
        ('sell', 'Sell'),
    ]
    
    SIGNAL_TYPE_CHOICES = [
        ('high_conviction', 'High Conviction ML'),
        ('harmonic', 'Harmonic Pattern'),
        ('quantum_mtf', 'Quantum Multi-Timeframe'),
        ('confluence', 'Confluence (2-Model)'),
        ('ultra', 'Ultra (3-Model Confluence)'),
        ('manual', 'Manual'),
    ]
    
    # Trade Identification
    id = models.AutoField(primary_key=True)
    signal_id = models.CharField(max_length=100, null=True, blank=True, db_index=True)
    pair = models.CharField(max_length=20, db_index=True)
    
    # Order Details
    order_type = models.CharField(max_length=10, choices=ORDER_TYPE_CHOICES)
    entry_price = models.DecimalField(max_digits=10, decimal_places=5)
    lot_size = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('0.01'))
    
    # Risk Management
    stop_loss = models.DecimalField(max_digits=10, decimal_places=5)
    take_profit_1 = models.DecimalField(max_digits=10, decimal_places=5, null=True, blank=True)
    take_profit_2 = models.DecimalField(max_digits=10, decimal_places=5, null=True, blank=True)
    take_profit_3 = models.DecimalField(max_digits=10, decimal_places=5, null=True, blank=True)
    
    # Execution Tracking
    entry_time = models.DateTimeField(default=timezone.now, db_index=True)
    exit_time = models.DateTimeField(null=True, blank=True)
    exit_price = models.DecimalField(max_digits=10, decimal_places=5, null=True, blank=True)
    
    # Performance Metrics
    pips_gained = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    profit_loss = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    risk_reward_ratio = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    
    # Status & Classification
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', db_index=True)
    signal_type = models.CharField(max_length=50, choices=SIGNAL_TYPE_CHOICES, null=True, blank=True)
    signal_source = models.CharField(max_length=50, null=True, blank=True)
    
    # Metadata
    notes = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'paper_trades'
        ordering = ['-entry_time']
        indexes = [
            models.Index(fields=['pair', 'status']),
            models.Index(fields=['entry_time', 'status']),
            models.Index(fields=['signal_type']),
        ]
    
    def __str__(self):
        return f"{self.pair} {self.order_type.upper()} @ {self.entry_price} ({self.status})"
    
    def calculate_pips(self):
        """Calculate pips gained/lost"""
        if not self.exit_price:
            return None
        
        pip_multiplier = 10000 if 'JPY' not in self.pair else 100
        
        if self.order_type == 'buy':
            pips = (float(self.exit_price) - float(self.entry_price)) * pip_multiplier
        else:  # sell
            pips = (float(self.entry_price) - float(self.exit_price)) * pip_multiplier
        
        return round(pips, 2)
    
    def calculate_profit_loss(self):
        """Calculate profit/loss in USD (simplified)"""
        pips = self.calculate_pips()
        if pips is None:
            return None
        
        # Simplified: $10 per pip for 1.0 lot, scale proportionally
        pip_value = 10 * float(self.lot_size)
        return round(pips * pip_value, 2)
    
    def close_trade(self, exit_price, exit_time=None):
        """Close the trade and calculate final metrics"""
        self.exit_price = Decimal(str(exit_price))
        self.exit_time = exit_time or timezone.now()
        self.pips_gained = self.calculate_pips()
        self.profit_loss = self.calculate_profit_loss()
        self.status = 'closed'
        self.save()
        
        return self.profit_loss


class PriceCache(models.Model):
    """Cache for price data from multiple sources"""
    
    TIMEFRAME_CHOICES = [
        ('1m', '1 Minute'),
        ('5m', '5 Minutes'),
        ('15m', '15 Minutes'),
        ('30m', '30 Minutes'),
        ('1h', '1 Hour'),
        ('4h', '4 Hours'),
        ('1d', '1 Day'),
        ('1w', '1 Week'),
    ]
    
    symbol = models.CharField(max_length=20, db_index=True)
    timestamp = models.DateTimeField(db_index=True)
    
    # OHLC Data
    open = models.DecimalField(max_digits=10, decimal_places=5)
    high = models.DecimalField(max_digits=10, decimal_places=5)
    low = models.DecimalField(max_digits=10, decimal_places=5)
    close = models.DecimalField(max_digits=10, decimal_places=5)
    volume = models.BigIntegerField(default=0)
    
    # Metadata
    source = models.CharField(max_length=50)  # alpha_vantage, twelve_data, etc.
    timeframe = models.CharField(max_length=10, choices=TIMEFRAME_CHOICES, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'price_cache'
        ordering = ['-timestamp']
        unique_together = ['symbol', 'timestamp', 'timeframe', 'source']
        indexes = [
            models.Index(fields=['symbol', '-timestamp']),
            models.Index(fields=['symbol', 'timeframe', '-timestamp']),
        ]
    
    def __str__(self):
        return f"{self.symbol} {self.timeframe} @ {self.timestamp}"


class PerformanceMetrics(models.Model):
    """Aggregate performance metrics by day/pair"""
    
    date = models.DateField(db_index=True)
    pair = models.CharField(max_length=20, db_index=True)
    
    # Trade Statistics
    total_trades = models.IntegerField(default=0)
    winning_trades = models.IntegerField(default=0)
    losing_trades = models.IntegerField(default=0)
    win_rate = models.DecimalField(max_digits=5, decimal_places=2, default=Decimal('0.00'))
    
    # Performance Metrics
    total_pips = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('0.00'))
    total_pnl = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('0.00'))
    avg_risk_reward = models.DecimalField(max_digits=5, decimal_places=2, default=Decimal('0.00'))
    max_drawdown = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('0.00'))
    
    # Equity Tracking
    starting_equity = models.DecimalField(max_digits=15, decimal_places=2, default=Decimal('10000.00'))
    ending_equity = models.DecimalField(max_digits=15, decimal_places=2, default=Decimal('10000.00'))
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'performance_metrics'
        ordering = ['-date']
        unique_together = ['date', 'pair']
        indexes = [
            models.Index(fields=['date', 'pair']),
            models.Index(fields=['-date']),
        ]
    
    def __str__(self):
        return f"{self.pair} - {self.date}: {self.win_rate}% WR, {self.total_pips} pips"
    
    def update_metrics(self):
        """Recalculate metrics based on closed trades"""
        from django.db.models import Sum, Avg, Count
        
        trades = PaperTrade.objects.filter(
            pair=self.pair,
            entry_time__date=self.date,
            status='closed'
        )
        
        self.total_trades = trades.count()
        self.winning_trades = trades.filter(pips_gained__gt=0).count()
        self.losing_trades = trades.filter(pips_gained__lt=0).count()
        
        if self.total_trades > 0:
            self.win_rate = (self.winning_trades / self.total_trades) * 100
        
        agg = trades.aggregate(
            total_pips=Sum('pips_gained'),
            total_pnl=Sum('profit_loss'),
            avg_rr=Avg('risk_reward_ratio')
        )
        
        self.total_pips = agg['total_pips'] or Decimal('0.00')
        self.total_pnl = agg['total_pnl'] or Decimal('0.00')
        self.avg_risk_reward = agg['avg_rr'] or Decimal('0.00')
        
        self.ending_equity = self.starting_equity + self.total_pnl
        
        self.save()


class APIUsageTracker(models.Model):
    """Track API usage to stay within free tier limits"""
    
    api_name = models.CharField(max_length=50, db_index=True)
    date = models.DateField(default=timezone.now, db_index=True)
    requests_made = models.IntegerField(default=0)
    requests_limit = models.IntegerField()
    
    # Rate limiting
    last_request_time = models.DateTimeField(null=True, blank=True)
    rate_limit_per_minute = models.IntegerField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'api_usage_tracker'
        unique_together = ['api_name', 'date']
        indexes = [
            models.Index(fields=['api_name', '-date']),
        ]
    
    def __str__(self):
        return f"{self.api_name} - {self.date}: {self.requests_made}/{self.requests_limit}"
    
    def can_make_request(self):
        """Check if we can make another request"""
        return self.requests_made < self.requests_limit
    
    def increment_usage(self):
        """Increment usage counter"""
        self.requests_made += 1
        self.last_request_time = timezone.now()
        self.save()
