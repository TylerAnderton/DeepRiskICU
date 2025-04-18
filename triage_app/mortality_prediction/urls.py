from django.urls import path
from . import views

app_name = 'mortality_prediction'
urlpatterns = [
    path('', views.mortality_prediction, name='mortality_prediction'),
]