from django.shortcuts import render, redirect
from django.contrib import messages
from utils.decorators import login_required_with_return
from accounts.models import Caregiver
from data_entry.models import Note, Prescription
from ml.pipeline import pipe

from .forms import PredictionForm

def predict_mortality_risk(
    patient,
    admission,
    notes,
    prescriptions
):
    patient_data = {
        'patient_id': patient.patient_id,
        'gender': patient.gender,
        'dob': patient.dob,
        'ethnicity': patient.ethnicity,
    }

    admission_data = {
        'admission_id': admission.admission_id,
        'admission_dt': admission.admission_dt,
        'admission_location': admission.admission_location,
        'insurance_type': admission.insurance_type,
    }

    notes_list = []
    for note in notes:
        notes_list.append({
            'note_id': note.note_id,
            'note_dt': note.note_dt,
            'note_text': note.note_text,
        })

    prescriptions_list = []
    for prescription in prescriptions:
        prescriptions_list.append({
            'rx_id': prescription.rx_id,
            'rx_start_date': prescription.rx_start_date,
            'rx_name': prescription.rx_name,
            'rx_dose': prescription.rx_dose,
        })

    print()
    print('<---- Pipeline Inputs ---->')
    print(f'patient_data: {patient_data}')
    print(f'admission_data: {admission_data}')
    print(f'notes_list: {notes_list}')
    print(f'prescriptions_list: {prescriptions_list}')
    
    input = {
        'patient_input': patient_data,
        'admission_input': admission_data,
        'notes_input': notes_list,
        'prescriptions_input': prescriptions_list,
    }

    return pipe(input) 

@login_required_with_return
def mortality_prediction(request):
    """View for mortality risk prediction."""
    if request.method == 'POST':
        try:
            caregiver = Caregiver.objects.get(user=request.user)
        except Caregiver.DoesNotExist:
            messages.error(request, "Error: Your user account is not associated with a caregiver profile.")
            return redirect('data_entry:new_note')
        
        prediction_form = PredictionForm(request.POST, caregiver=caregiver)
        
        if prediction_form.is_valid():
            patient = prediction_form.cleaned_data['patient']
            admission = prediction_form.cleaned_data['admission']

            notes = Note.objects.filter(
                patient=patient,
                admission=admission
            )
            prescriptions = Prescription.objects.filter(
                patient=patient,
                admission=admission
            )
            
            prediction = prediction_form.save(commit=False)
            prediction.caregiver = caregiver
            prediction.patient = patient
            prediction.admission = admission
            
            prediction_val = predict_mortality_risk(
                patient=patient,
                admission=admission,
                notes=notes,
                prescriptions=prescriptions
            )

            prediction.mortality_risk = prediction_val
            prediction.save()
            
            prediction.notes.set(notes)
            prediction.prescriptions.set(prescriptions)
            
        else: # form invalid
            messages.error(request, "Please provide both Subject ID and Admission ID")

    else: # GET request
        prediction_form = PredictionForm()
        prediction = None
        
    context = {
        'prediction_form': prediction_form,
        'prediction': prediction,
    }
    
    return render(
        request,
        'mortality_prediction/mortality_prediction.html',
        context
    )
