"""
Paper Trading Package
Enterprise-level paper trading system for forex
"""

# Don't import models/engine at module level to avoid AppRegistryNotReady
# Import them when needed in your code instead

default_app_config = 'paper_trading.apps.PaperTradingConfig'

__all__ = [
    'PaperTradingEngine',
    'DataAggregator',
    'PaperTrade',
    'PriceCache',
    'PerformanceMetrics',
    'APIUsageTracker',
]
