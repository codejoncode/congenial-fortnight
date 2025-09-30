from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'pairs', views.ForexPairViewSet)
router.register(r'signals', views.SignalViewSet)
router.register(r'trades', views.TradeViewSet)

urlpatterns = [
    path('', include(router.urls)),
]