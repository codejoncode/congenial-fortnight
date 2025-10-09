from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'signals', views.SignalViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('backtest/', views.backtest_results, name='backtest_results'),
    path('backtest/csv/', views.download_backtest_csv, name='download_backtest_csv'),
    path('historical/', views.get_historical_data, name='get_historical_data'),
    path('health/', views.health_check, name='health_check'),
    path('unified/', views.unified_signals, name='unified_signals'),
    path('holloway/<str:pair>/', views.get_holloway, name='get_holloway'),
    path('signals/<str:pair>/', views.get_signals, name='get_signals'),
    path('backtest/<str:pair>/', views.trading_backtest, name='trading_backtest'),
    path('data/status/', views.data_status, name='data_status'),
    path('data/update/', views.update_data, name='update_data'),
]