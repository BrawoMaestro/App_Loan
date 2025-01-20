from django.urls import path
from . import views
from .views import loan_prediction

urlpatterns = [
    path('', loan_prediction, name='loan_prediction'),
    path('feature_importance/', views.feature_importance, name='feature_importance'),
]
