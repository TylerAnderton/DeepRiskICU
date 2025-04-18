from django import forms
from django.forms import DateInput, DateTimeInput

from .models import Note, Patient, Admission, Prescription
from data_entry.widgets import PatientWidget

class PatientForm(forms.Form):  # Change from ModelForm to Form to allow for updating existing patients
    # patient_id = forms.IntegerField(label="Patient ID")
    first_name = forms.CharField(max_length=50, label="First Name")
    last_name = forms.CharField(max_length=50, label="Last Name")
    gender = forms.ChoiceField(choices=Patient.GENDERS)
    dob = forms.DateField(widget=DateInput(attrs={'type': 'date'}))
    ethnicity = forms.ChoiceField(choices=Patient.ETHNICITIES)

    def save(self):
        """Create or update a Patient instance based on form data"""
        return Patient.objects.create(
            # patient_id=self.cleaned_data['patient_id'],
            first_name=self.cleaned_data['first_name'],
            last_name=self.cleaned_data['last_name'],
            gender=self.cleaned_data['gender'],
            dob=self.cleaned_data['dob'],
            ethnicity=self.cleaned_data['ethnicity']
        )

class AdmissionForm(forms.ModelForm):
    class Meta:
        model = Admission
        fields = [
            'patient',
            'admission_dt',
            'admission_location',
            'insurance_type',
            # caregiver set automatically
        ]
        widgets = {
            'patient': PatientWidget(attrs={'style': 'width: 30%;'}),
            'admission_dt': DateTimeInput(attrs={'type': 'datetime-local'}),
        }

    def __init__(self, *args, **kwargs):
        # Extract the caregiver from the kwargs before calling parent constructor
        self.caregiver = kwargs.pop('caregiver', None)
        super().__init__(*args, **kwargs)
        
    def save(self, commit=True):
        instance = super().save(commit=False)
        
        if commit:
            instance.save()
            instance.caregiver.add(self.caregiver)
            self.save()
            
        return instance

    
class PrescriptionForm(forms.ModelForm):
    class Meta:
        model = Prescription
        fields = [
            'patient',
            'admission',
            'rx_start_date',
            # 'rx_end_date', # left out for now
            'rx_name',
            'rx_dose',
        ]
        widgets = {
            'patient': PatientWidget(attrs={'style': 'width: 30%;'}),
            'admission': forms.Select(attrs={'style': 'width: 30%;'}),
            'rx_start_date': DateInput(attrs={'type': 'date'}),
        }

    def __init__(self, *args, **kwargs):
        # Extract the caregiver from the kwargs before calling parent constructor
        self.caregiver = kwargs.pop('caregiver', None)
        super().__init__(*args, **kwargs)
        
        self.fields['admission'].queryset = Admission.objects.none()
        
        if self.instance.pk and self.instance.patient:
            self.fields['admission'].queryset = Admission.objects.filter(patient=self.instance.patient)
            
        if 'patient' in self.data:
            try:
                patient_id = int(self.data.get('patient'))
                self.fields['admission'].queryset = Admission.objects.filter(patient=patient_id)
            except (ValueError, TypeError):
                pass
        
    def save(self, commit=True):
        instance = super().save(commit=False)
        
        if commit:
            instance.save()
            instance.caregiver.add(self.caregiver)
            self.save()
            
        return instance

class NoteForm(forms.ModelForm):
    class Meta:
        model = Note
        fields = [
            'patient',
            'admission',
            'note_dt',
            'note_text'
        ]
        widgets = {
            'patient': PatientWidget(attrs={'style': 'width: 30%;'}),
            'admission': forms.Select(attrs={'style': 'width: 30%;'}),
            'note_dt': DateTimeInput(attrs={'type': 'datetime-local'}), 
        }

    def __init__(self, *args, **kwargs):
        # Extract the caregiver from the kwargs before calling parent constructor
        self.caregiver = kwargs.pop('caregiver', None)
        super().__init__(*args, **kwargs)
        
        self.fields['admission'].queryset = Admission.objects.none()
        
        if self.instance.pk and self.instance.patient:
            self.fields['admission'].queryset = Admission.objects.filter(patient=self.instance.patient)
            
        if 'patient' in self.data:
            try:
                patient_id = int(self.data.get('patient'))
                self.fields['admission'].queryset = Admission.objects.filter(patient=patient_id)
            except (ValueError, TypeError):
                pass
        
    def save(self, commit=True):
        instance = super().save(commit=False)
        
        if commit:
            instance.save()
            instance.caregiver.add(self.caregiver)
            
        return instance
        