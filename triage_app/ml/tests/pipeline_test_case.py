import numpy as np
import datetime
import os
import joblib
from django.conf import settings

from unittest import TestCase
from unittest.mock import MagicMock

ONE_HOT_ENCODER_PATH = settings.ONE_HOT_ENCODER_PATH
EMBEDDINGS_LENGTH = 768

class MLTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up data used in all test methods"""
        super().setUpClass()
        cls.sample_input ={
            'patient_input': {
                'patient_id': 1000,
                'gender': 'male',
                'dob': datetime.date(1969, 3, 1),
                'ethnicity': 'hispanic'
            },

            'admission_input': {
                'admission_id': 1000,
                'admission_dt': datetime.datetime(2025, 3, 1, 10, 15, tzinfo=datetime.timezone.utc),
                'admission_location': 'er',
                'insurance_type': 'medicare'
            },

            'notes_input': [
                {
                    'note_id': 1,
                    'note_dt': datetime.datetime(2025, 4, 3, 3, 55, tzinfo=datetime.timezone.utc),
                    'note_text': 'Patient complained of headaches'    
                },
                {
                    'note_id': 2,
                    'note_dt': datetime.datetime(2025, 3, 13, 23, 38, tzinfo=datetime.timezone.utc),
                    'note_text': 'Patient fell out of bed. Assessed to have minor contusion on right hip.'
                },
                {
                    'note_id': 3,
                    'note_dt': datetime.datetime(2025, 3, 12, 23, 40, tzinfo=datetime.timezone.utc),
                    'note_text': "Patient said he doesn't like his medications"
                }
            ],

            'prescriptions_input': [
                {
                    'rx_id': 1,
                    'rx_start_date': datetime.date(2025, 3, 8),
                    'rx_name': 'Trazadone',
                    'rx_dose': '100 mg'
                },
                {
                    'rx_id': 2,
                    'rx_start_date': datetime.date(2025, 3, 8),
                    'rx_name': 'Modafiniil',
                    'rx_dose': '100 mg'
                }
            ]
        } 
        
        patient_id = cls.sample_input['patient_input']['patient_id']
        cls.sample_input['admission_input']['patient_id'] = patient_id

        for note in cls.sample_input['notes_input']:
            note['admission_id'] = cls.sample_input['admission_input']['admission_id']
            
        for prescription in cls.sample_input['prescriptions_input']:
            prescription['admission_id'] = cls.sample_input['admission_input']['admission_id']
        
        # # Mock encoder for testing
        # cls.mock_encoder = MagicMock()

        # real_encoder = joblib.load(ONE_HOT_ENCODER_PATH)
        # feature_names = list(real_encoder.get_feature_names_out())

        # cls.mock_encoder.get_feature_names_out.return_value = feature_names
        # cls.mock_encoder.transform.return_value = np.ones((1, len(feature_names)))
        # print(f'Mock encoder feature names: {feature_names}')
        # print(f'Mock encoder return values: {cls.mock_encoder.transform.return_value}')
        
        # # Mock embeddings
        # cls.mock_embedding = np.ones(768)  # Standard size for ClinicalBERT embeddings
