import numpy as np
from unittest.mock import patch

from ..pipeline import get_patient_admission_data, calculate_ages
from ..embedded_features import get_notes_data, concatenate_notes, get_note_embeddings
from ..embedded_features import get_prescriptions_data, concatenate_prescriptions, get_prescription_embeddings
from .pipeline_test_case import MLTestCase, EMBEDDINGS_LENGTH

class EmbeddedFeaturesTestCase(MLTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    # @patch('ml.embedded_features.get_bert_embeddings')
    def test_note_embeddings(self):
        """Test note embedding generation"""
        # mock_get_embeddings.return_value = np.array([self.mock_embedding])
        
        patient_admission_data = get_patient_admission_data(self.sample_input)
        patient_admission_data = calculate_ages(patient_admission_data)

        len_notes = len(self.sample_input['notes_input'])
        
        notes_data = get_notes_data(self.sample_input, patient_admission_data)
        self.assertEqual(notes_data.shape[0], len_notes) # One row per note
        
        notes_concat = concatenate_notes(notes_data)
        self.assertEqual(notes_concat.shape[0], 1) # One row per admission
        
        result = get_note_embeddings(notes_concat)
        self.assertEqual(result.shape[0], 1)
        
        self.assertIsInstance(result['note_embedding'].iloc[0], np.ndarray)
        self.assertEqual(result['note_embedding'].iloc[0].shape[0], EMBEDDINGS_LENGTH)
        
    # @patch('ml.embedded_features.get_bert_embeddings')
    def test_prescription_embeddings(self):
        """Test prescription embedding generation"""
        # mock_get_embeddings.return_value = np.array([self.mock_embedding])
        
        len_prescriptions = len(self.sample_input['prescriptions_input'])
        
        prescription_data = get_prescriptions_data(self.sample_input)
        self.assertEqual(prescription_data.shape[0], len_prescriptions) # One row per prescription
        
        prescriptions_concat = concatenate_prescriptions(prescription_data)
        self.assertEqual(prescriptions_concat.shape[0], 1) # One row per admission
        
        result = get_prescription_embeddings(prescriptions_concat)
        self.assertEqual(result.shape[0], 1)
        
        self.assertIsInstance(result['prescription_embedding'].iloc[0], np.ndarray)
        self.assertEqual(result['prescription_embedding'].iloc[0].shape[0], EMBEDDINGS_LENGTH)