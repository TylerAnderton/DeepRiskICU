from django import forms
from data_entry.models import Note, Patient, Admission, Prescription
from data_entry.widgets import PatientWidget
from .models import MortalityPrediction

class PredictionForm(forms.ModelForm):
    class Meta:
        model = MortalityPrediction
        fields = [
            'patient',
            'admission',
            'notes',
            'prescriptions',
        ]
        widgets = {
            'patient': PatientWidget(attrs={'style': 'width: 30%;'}),
            'admission': forms.Select(attrs={'style': 'width: 30%;'}),
        }

    def __init__(self, *args, **kwargs):
        # Extract the caregiver from the kwargs before calling parent constructor
        self.caregiver = kwargs.pop('caregiver', None)
        super().__init__(*args, **kwargs)
        
        self.fields['notes'].required = False
        self.fields['prescriptions'].required = False

        self.fields['notes'].queryset = Note.objects.none()
        self.fields['prescriptions'].queryset = Prescription.objects.none()
        self.fields['admission'].queryset = Admission.objects.none()
        
        if self.instance.pk and self.instance.patient:
            self.fields['admission'].queryset = Admission.objects.filter(patient=self.instance.patient)
            
        if 'patient' in self.data:
            try:
                patient = int(self.data.get('patient'))
                self.fields['admission'].queryset = Admission.objects.filter(patient=patient)
                
                # if 'admission' in self.data:
                #     admission = int(self.data.get('admission'))

                #     self.fields['notes'].queryset = Note.objects.filter(
                #         patient=patient,
                #         admission=admission
                #     )
                #     self.fields['prescriptions'].queryset = Prescription.objects.filter(
                #         patient=patient,
                #         admission=admission
                #     )

            except (ValueError, TypeError):
                pass
        
    def save(self, commit=True):
        instance = super().save(commit=False)
        
        if commit:
            instance.save()
            instance.caregiver.add(self.caregiver)
            self.save()
            
        return instance