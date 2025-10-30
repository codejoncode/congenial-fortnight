"""
Django app configuration for paper trading
"""
from django.apps import AppConfig


class PaperTradingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'paper_trading'
    verbose_name = 'Paper Trading System'
    
    def ready(self):
        """
        Application initialization
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info('📊 Paper Trading System initialized')
