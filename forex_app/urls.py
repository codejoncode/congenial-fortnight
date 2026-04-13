from django.urls import path
from . import views

urlpatterns = [
    path('api/generate-signal/', views.generate_signal, name='generate_signal'),
    path('api/update-data/', views.update_data, name='forex_app_update_data'),
    path('api/backtest/', views.run_backtest, name='backtest'),
]
