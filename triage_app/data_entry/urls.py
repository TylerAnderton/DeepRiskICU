from django.urls import path
from . import views

app_name = 'data_entry'
urlpatterns = [
    path('', views.EntrySelectionView.as_view(), name='entry_selection'),
    path('patient/', views.new_patient, name='new_patient'),
    path('admission/', views.new_admission, name='new_admission'),
    path('prescription/', views.new_prescription, name='new_prescription'),
    path('note/', views.new_note, name='new_note'), 
]