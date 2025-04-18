document.addEventListener('DOMContentLoaded', function() {
    // // Get the patient select element (using jQuery for Select2 compatibility)
    // const patientSelect = $('#id_patient');
    // // Get the admission select element (regular DOM element)
    // const admissionSelect = document.getElementById('id_admission');
    
    // Support configurable selectors and endpoint
    const config = window.admissionsDropdownConfig || {};
    
    // Get selectors with fallbacks
    const patientSelectSelector = config.patientSelector || '#id_patient';
    const admissionSelectSelector = config.admissionSelector || '#id_admission';
    
    // API endpoint with fallback
    const apiEndpoint = config.apiEndpoint || '/api/get_admissions/';
    
    // Get the patient select element (using jQuery for Select2 compatibility)
    const patientSelect = $(patientSelectSelector);
    // Get the admission select element (regular DOM element)
    const admissionSelect = document.querySelector(admissionSelectSelector);
    
    if (patientSelect.length && admissionSelect) {
        // Function to update admission options
        function updateAdmissionOptions(patientId) {
            // Clear current options
            admissionSelect.innerHTML = '<option value="">---------</option>';
            
            if (!patientId) return;
            
            console.log('Fetching admissions for patient:', patientId);
            
            // Fetch admissions for the selected patient
            fetch(`${apiEndpoint}?patient=${patientId}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('Received data:', data); // Debug logging
                    
                    // Handle Select2 formatted response
                    if (data.results && data.results.length > 0) {
                        data.results.forEach(admission => {
                            const option = document.createElement('option');
                            option.value = admission.id;
                            option.textContent = admission.text;
                            admissionSelect.appendChild(option);
                        });
                        console.log('Added options to admission dropdown');
                    } else {
                        console.log('No admissions found for this patient');
                    }
                })
                .catch(error => {
                    console.error('Error fetching admissions:', error);
                });
        }
        
        // Set up event listener for Select2 changes
        patientSelect.on('select2:select', function() {
            const patientId = $(this).val();
            console.log('Patient changed to:', patientId); // Debug logging
            updateAdmissionOptions(patientId);
        });
        
        // Initialize admissions if patient is already selected
        setTimeout(function() {
            if (patientSelect.val()) {
                console.log('Initial patient value:', patientSelect.val());
                updateAdmissionOptions(patientSelect.val());
            }
        }, 500); // Small delay to ensure Select2 is fully initialized
    } else {
        console.log('Could not find patient or admission select elements');
    }
});