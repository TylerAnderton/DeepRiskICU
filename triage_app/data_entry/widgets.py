from django_select2 import forms as s2forms

class PatientWidget(s2forms.ModelSelect2Widget):
    search_fields = ['patient_id__istartswith']
    max_results = 10
    minimum_input_length = 0
    
    def label_from_instance(self, obj):
        return str(obj.patient_id)
    
# class AdmissionWidget(s2forms.ModelSelect2Widget):
#     search_fields = ['admission_id__istartswith']
#     max_results = 10
#     minimum_input_length = 0
   
#     def label_from_instance(self, obj):
#         return str(obj.admission_id)