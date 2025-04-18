from django.db import models
from accounts.models import Caregiver
from data_entry.models import Patient, Admission, Prescription, Note

class MortalityPrediction(models.Model):
    prediction_id = models.AutoField(primary_key=True)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='mortality_predictions')
    admission = models.ForeignKey(Admission, on_delete=models.CASCADE, related_name='mortality_predictions')
    caregiver = models.ForeignKey(Caregiver, on_delete=models.SET_NULL, null=True, related_name='mortality_predictions')
    notes = models.ManyToManyField(Note, related_name='mortality_predictions')
    prescriptions = models.ManyToManyField(Prescription, related_name='mortality_predictions')
    mortality_risk = models.FloatField(null=True)
