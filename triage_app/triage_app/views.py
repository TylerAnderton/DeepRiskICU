from django.views.generic import TemplateView
from django.http import JsonResponse
from data_entry.models import Patient, Admission
from utils.decorators import login_required_with_return

class HomeView(TemplateView):
    template_name = 'home.html'
    
@login_required_with_return
def get_admissions(request):
    """AJAX view to get admissions for a specific patient"""
    patient_id = request.GET.get('patient')
    
    if patient_id:
        # Query admissions for this patient
        admissions = Admission.objects.filter(patient_id=patient_id)
        print(f"Found {admissions.count()} admissions for patient {patient_id}")
        
        # Format the response in Select2-compatible format
        results = [
            {"id": admission.admission_id, "text": str(admission.admission_id)}
            for admission in admissions
        ]
        
        return JsonResponse({"results": results, "more": False})
    
    return JsonResponse({"results": [], "more": False})