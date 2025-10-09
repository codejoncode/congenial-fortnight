"""
URL routing for paper trading API
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create router for ViewSets
router = DefaultRouter()
router.register(r'trades', views.PaperTradeViewSet, basename='paper-trade')
router.register(r'metrics', views.PerformanceMetricsViewSet, basename='performance-metrics')

# URL patterns
urlpatterns = [
    # Router URLs
    path('', include(router.urls)),
    
    # Price data endpoints
    path('price/realtime/', views.get_realtime_price, name='realtime-price'),
    path('price/ohlc/', views.get_historical_ohlc, name='historical-ohlc'),
    
    # Position management
    path('positions/update/', views.update_positions, name='update-positions'),
    
    # MetaTrader endpoints
    path('mt/account/', views.mt_account_info, name='mt-account'),
    path('mt/positions/', views.mt_positions, name='mt-positions'),
]
