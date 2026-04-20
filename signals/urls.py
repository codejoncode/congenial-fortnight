from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Router for the SignalViewSet (database signals list)
router = DefaultRouter()
router.register(r'db', views.SignalViewSet, basename='signal-db')

urlpatterns = [
    # Router URLs (creates /api/db/ endpoint for Signal model)
    path('', include(router.urls)),
    
    # Custom signal endpoints — fixed paths MUST come before <str:pair> wildcard
    path('signals/', views.SignalViewSet.as_view({'get': 'list'}), name='signals_list'),
    path('signals/unified/', views.unified_signals, name='unified_signals'),
    path('signals/generate/', views.generate_signals, name='generate_signals'),
    path('signals/<str:pair>/', views.get_signals, name='get_signals'),   # keep last

    # Backtest endpoints — fixed paths before wildcard
    path('backtest/', views.backtest_results, name='backtest_results'),
    path('backtest/csv/', views.download_backtest_csv, name='download_backtest_csv'),
    path('backtest/<str:pair>/', views.trading_backtest, name='trading_backtest'),
    
    # Data endpoints
    path('historical/', views.get_historical_data, name='get_historical_data'),
    path('holloway/<str:pair>/', views.get_holloway, name='get_holloway'),
    path('data/status/', views.data_status, name='data_status'),
    path('data/update/', views.update_data, name='update_data'),
    path('data/update-all/', views.update_data, name='update_data_all'),
    
    # System health + decision engine
    path('health/', views.health_check, name='api_health_check'),
    path('system-health/', views.system_health, name='system_health'),
    path('signals/decision/', views.signal_decision, name='signal_decision'),
    path('signals/decision/<str:pair>/', views.signal_decision, name='signal_decision_pair'),

    path('paper-trades/execute/', views.execute_paper_trade, name='execute_paper_trade'),

    # Signal performance / accuracy endpoint
    path('signal-performance/', views.signal_performance, name='signal_performance'),
]