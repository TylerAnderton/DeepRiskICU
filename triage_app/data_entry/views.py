from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.generic import TemplateView
# from django.forms import ModelForm, DateInput, DateTimeInput
from django.utils.decorators import method_decorator

from datetime import datetime

from accounts.models import Caregiver
from utils.decorators import login_required_with_return
from .forms import PatientForm, AdmissionForm, PrescriptionForm, NoteForm

# Entry selection page
class EntrySelectionView(TemplateView):
    template_name = 'data_entry/entry_selection.html'

    @method_decorator(login_required_with_return)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
@login_required_with_return
def new_patient(request):
    """View for handling new patient data entry."""
    if request.method == 'POST':
        try:
            caregiver = Caregiver.objects.get(user=request.user)
        except Caregiver.DoesNotExist:
            messages.error(request, "Error: Your user account is not associated with a caregiver profile.")
            return redirect('data_entry:new_patient')

        patient_form = PatientForm(request.POST)
        if patient_form.is_valid():
            patient = patient_form.save()
            messages.info(request, f"New patient profile created with ID {patient.patient_id}.")
            
            messages.success(request, "Patient profile saved successfully!")
            return redirect('data_entry:entry_selection')

        else:
            messages.error(request, "Please correct the errors in the form.")

    else:
        patient_form = PatientForm()
    
    context = {
        'patient_form': patient_form,
    }
    
    return render(request, 'data_entry/patient.html', context)

@login_required_with_return
def new_admission(request):
    """View for handling new patient admission data entry."""
    if request.method == 'POST':
        try:
            caregiver = Caregiver.objects.get(user=request.user)
        except Caregiver.DoesNotExist:
            messages.error(request, "Error: Your user account is not associated with a caregiver profile.")
            return redirect('data_entry:new_admission')

        admission_form = AdmissionForm(request.POST, caregiver=caregiver)

        if admission_form.is_valid():
            patient = admission_form.cleaned_data['patient']
            patient_id = patient.patient_id

            admission = admission_form.save(commit=False)
            admission.patient = patient
            admission.caregiver = caregiver
            admission.save()
            
            messages.success(request, f'New admission created for patient {patient_id} with ID {admission.admission_id}.')
            return redirect('data_entry:entry_selection')

        else:
            messages.error(request, "Please correct the errors in the form.")

    else:
        admission_form = AdmissionForm(initial={'admission_dt': datetime.now().strftime('%Y-%m-%dT%H:%M')},)
    
    context = {
        'admission_form': admission_form,
    }
    
    return render(request, 'data_entry/admission.html', context)
    
@login_required_with_return
def new_prescription(request):
    """View for handling data entry form submissions."""
    if request.method == 'POST':
        try:
            caregiver = Caregiver.objects.get(user=request.user)
        except Caregiver.DoesNotExist:
            messages.error(request, "Error: Your user account is not associated with a caregiver profile.")
            return redirect('data_entry:new_prescription')

        prescription_form = PrescriptionForm(request.POST, caregiver=caregiver)

        if prescription_form.is_valid():
            patient = prescription_form.cleaned_data['patient']
            admission = prescription_form.cleaned_data['admission']
            
            rx_start_date = prescription_form.cleaned_data['rx_start_date']
            if rx_start_date < datetime.date(admission.admission_dt) or (admission.discharge_dt and rx_start_date > datetime.date(admission.discharge_dt)):
                messages.error(request, f"Error: Prescription date {rx_start_date} is not between the admission dates.")
                return redirect('data_entry:new_prescription')

            prescription = prescription_form.save(commit=False)
            prescription.caregiver = caregiver
            prescription.patient = patient
            prescription.admission = admission
            prescription.save()
            
            messages.success(request, "Prescripion saved successfully!")
            return redirect('data_entry:entry_selection')

        else:
            messages.error(request, "Please correct the errors in the form.")

    else:
        prescription_form = PrescriptionForm()
    
    context = {
        'prescription_form': prescription_form,
    }
    
    return render(request, 'data_entry/prescription.html', context)

@login_required_with_return
def new_note(request):
    """View for handling data entry form submissions."""
    if request.method == 'POST':
        try:
            caregiver = Caregiver.objects.get(user=request.user)
        except Caregiver.DoesNotExist:
            messages.error(request, "Error: Your user account is not associated with a caregiver profile.")
            return redirect('data_entry:new_note')

        note_form = NoteForm(request.POST, caregiver=caregiver)

        if note_form.is_valid():
            patient = note_form.cleaned_data['patient']
            admission = note_form.cleaned_data['admission']
            
            note_dt = note_form.cleaned_data['note_dt']
            if note_dt < admission.admission_dt or (admission.discharge_dt and note_dt > admission.discharge_dt):
                messages.error(request, f"Error: Note datetime {note_dt} is not between the admission dates.")
                return redirect('data_entry:new_note')

            note = note_form.save(commit=False)
            note.caregiver = caregiver
            note.patient = patient
            note.admission = admission
            note.save()
            
            messages.success(request, "Note saved successfully!")
            return redirect('data_entry:entry_selection')

        else:
            messages.error(request, "Please correct the errors in the form.")

    else:
        note_form = NoteForm(initial={'note_dt': datetime.now().strftime('%Y-%m-%dT%H:%M')},)
    
    context = {
        'note_form': note_form,
    }
    
    return render(request, 'data_entry/note.html', context)