from django.db import models
from django.contrib.auth.models import User

class Caregiver(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    # TODO: Make sure caregiver_id is automatically filled in when submitting data
    caregiver_id = models.AutoField(primary_key=True)
    
    # Caregiver groups
    NURSE = 'nurse'
    PHYSICIAN = 'physician'
    IMAGING = 'imaging'
    NUTRITION = 'nutrition'
    REHAB = 'rehab'
    PHARMACY = 'pharmacy'
    OTHER = 'other'
    CAREGIVER_GROUPS = [
        (NURSE, 'Nurse'),
        (PHYSICIAN, 'Physician'),
        (IMAGING, 'Imaging'),
        (NUTRITION, 'Nutrition'),
        (REHAB, 'Rehab'),
        (PHARMACY, 'Pharmacy'),
        (OTHER, 'Other'),
    ]
    caregiver_group = models.CharField(
        choices=CAREGIVER_GROUPS,
        null=False,
        max_length=10,
        default=OTHER,
    )
    # Could add additional caregiver information here
    
