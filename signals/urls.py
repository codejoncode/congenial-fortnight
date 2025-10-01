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
]