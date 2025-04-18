from django.db import models
from accounts.models import Caregiver

class Patient(models.Model):
    # Unique IDs
    patient_id = models.IntegerField(primary_key=True)
    
    # Name
    first_name = models.CharField(max_length=50, null=False, default='John')
    last_name = models.CharField(max_length=50, null=False, default='Doe')
    
    # Gender
    MALE = 'male'
    FEMALE = 'female'
    GENDERS = [
        (MALE, 'Male'),
        (FEMALE, 'Female'),
    ]
    gender = models.CharField(
        choices=GENDERS,
        null=False,
        max_length=10,
    )

    # Date of Birth/Death
    dob = models.DateField(null=False)
    dod = models.DateField(null=True)

    # Ethnicity
    WHITE = 'white'
    BLACK = 'black'
    HISPANIC = 'hispanic'
    ASIAN = 'asian'
    OTHER = 'other'
    ETHNICITIES = [
        (WHITE, 'White'),
        (BLACK, 'Black'),
        (HISPANIC, 'Hispanic'),
        (ASIAN, 'Asian'),
        (OTHER, 'Other'),
    ]
    ethnicity = models.CharField(
        choices=ETHNICITIES,
        null=False,
        max_length=10,
    )

    def save(self, *args, **kwargs):
        assert self.patient_id == None, "Patient ID should not be set manually"

        # Start patient IDs at 1000 to force 4+ digit IDs
        last_patient = Patient.objects.order_by('patient_id').last()
        self.patient_id = last_patient.patient_id + 1 if last_patient else 1000
        super().save(*args, **kwargs)
    
    def __str__(self):
        return str(self.patient_id)

class Admission(models.Model):
    # Unique IDs
    admission_id = models.AutoField(primary_key=True)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='admissions')
    caregiver = models.ForeignKey(Caregiver, on_delete=models.SET_NULL, null=True, related_name='admissions') 
    
    # Admission Datetime
    admission_dt = models.DateTimeField(null=False)
    discharge_dt = models.DateTimeField(null=True)

    # Admission Locations
    ER = 'er'
    PHYS_REF = 'phys_ref'
    CLINIC_REF = 'clinic_ref'
    TRANSFER = 'transfer'
    OTHER = 'other'
    ADMISSION_LOCATIONS = [
        (ER, 'ER'),
        (PHYS_REF, 'Physician Referral'),
        (CLINIC_REF, 'Clinic Referral'),
        (TRANSFER, 'Transfer'),
        (OTHER, 'Other'),
    ]
    admission_location = models.CharField(
        choices=ADMISSION_LOCATIONS,
        default=OTHER,
        max_length=10,
    )

    # Insurance Types
    PRIVATE = 'private'
    MEDICARE = 'medicare'
    MEDICAID = 'medicaid'
    SELF_PAY = 'self_pay'
    GOVERNMENT = 'government'
    OTHER = 'other'
    INSURANCE_TYPES = [
        (PRIVATE, 'Private'),
        (MEDICARE, 'Medicare'),
        (MEDICAID, 'Medicaid'),
        (SELF_PAY, 'Self Pay'),
        (GOVERNMENT, 'Government'),
        (OTHER, 'Other'),
    ]
    insurance_type = models.CharField(
        choices=INSURANCE_TYPES,
        default=SELF_PAY,
        max_length=10,
    )
    
    def save(self, *args, **kwargs):
        assert self.admission_id == None, "Admission ID should not be set manually"

        # Start admission IDs at 1000 to force 4+ digit IDs
        last_admission = Admission.objects.order_by('admission_id').last()
        self.admission_id = last_admission.admission_id + 1 if last_admission else 1000
        super().save(*args, **kwargs)

    def __str__(self):
        return str(self.admission_id)

class Note(models.Model):
    note_id = models.AutoField(primary_key=True)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='notes')
    admission = models.ForeignKey(Admission, on_delete=models.CASCADE, related_name='notes')
    note_dt = models.DateTimeField(null=False)
    caregiver = models.ForeignKey(Caregiver, on_delete=models.SET_NULL, null=True, related_name='notes')
    note_text = models.TextField(null=False)

class Prescription(models.Model):
    rx_id = models.AutoField(primary_key=True)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='prescriptions')
    admission = models.ForeignKey(Admission, on_delete=models.CASCADE, related_name='prescriptions')
    rx_start_date = models.DateField(null=False)
    rx_end_date = models.DateField(null=True)
    rx_name = models.CharField(max_length=100, null=False)
    rx_dose = models.CharField(max_length=100, null=False)