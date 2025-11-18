from django.urls import path
from .api import views as api_views

urlpatterns = [
    path('api/update-data/', api_views.update_data_api, name='update_data_api'),
    path('api/generate-signal/', api_views.generate_signal_api, name='generate_signal_api'),
]
