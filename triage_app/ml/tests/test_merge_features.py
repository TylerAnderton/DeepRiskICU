import numpy as np
import pandas as pd

from ..pipeline import get_patient_admission_data, calculate_ages, encode_categorical_features
from ..embedded_features import get_notes_data, concatenate_notes, get_note_embeddings
from ..embedded_features import get_prescriptions_data, concatenate_prescriptions, get_prescription_embeddings
from ..merge_features import merge_features, fill_missing_embeddings
from .pipeline_test_case import MLTestCase, EMBEDDINGS_LENGTH

class MergeFeaturesTestCase(MLTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_merge_features(self):
        """Test merging of different feature types"""
        # Create sample DataFrames to merge
        # patient_features = pd.DataFrame({
        #     'admission_id': [1000],
        #     'age_yrs': [56.0],
        #     'gender_male': [1.0]
        # })
        
        # note_features = pd.DataFrame({
        #     'admission_id': [1000],
        #     'note_embedding': [self.mock_embedding]
        # })
        
        # rx_features = pd.DataFrame({
        #     'admission_id': [1000],
        #     'prescription_embedding': [self.mock_embedding]
        # })

        patient_admission_data = get_patient_admission_data(self.sample_input)
        patient_admission_data = calculate_ages(patient_admission_data)
        patient_admission_features = encode_categorical_features(patient_admission_data, one_hot_encoder_path)

        notes_data = get_notes_data(self.sample_input, patient_admission_data)
        notes_concat = concatenate_notes(notes_data)
        note_features = get_note_embeddings(notes_concat)

        prescriptions_data = get_prescriptions_data(self.sample_input)
        prescriptions_concat = concatenate_prescriptions(prescriptions_data)
        prescription_features = get_prescription_embeddings(prescriptions_concat)
        
        result = merge_features(patient_admission_features, note_features, prescription_features)
        
        self.assertEqual(result.shape[0], 1)
        self.assertIn('age_yrs', result.columns)
        self.assertIn('note_embedding', result.columns)
        self.assertIn('prescription_embedding', result.columns)
        self.assertNotIn('admission_id', result.columns)  # Should be dropped
        
    def test_fill_missing_embeddings(self):
        """Test filling missing embeddings with zeros"""
        # Create DataFrame with missing embedding
        # df = pd.DataFrame({
        #     'age_yrs': [56.0],
        #     'gender_male': [1.0],
        #     'note_embedding': [self.mock_embedding],
        #     'prescription_embedding': [None]
        # })
        
        input = self.sample_input
        input['notes_input'] = []
        
        patient_admission_data = get_patient_admission_data(input)
        patient_admission_data = calculate_ages(patient_admission_data)
        patient_admission_features = encode_categorical_features(patient_admission_data, one_hot_encoder_path)

        notes_data = get_notes_data(input, patient_admission_data)
        notes_concat = concatenate_notes(notes_data)
        note_features = get_note_embeddings(notes_concat)

        prescriptions_data = get_prescriptions_data(input)
        prescriptions_concat = concatenate_prescriptions(prescriptions_data)
        prescription_features = get_prescription_embeddings(prescriptions_concat)

        merged_features = merge_features(
            patient_admission_features,
            note_features,
            prescription_features
        )
        result = fill_missing_embeddings(merged_features)
        
        self.assertTrue(isinstance(result['prescription_embedding'].iloc[0], np.ndarray))
        self.assertEqual(result['prescription_embedding'].iloc[0].shape[0], EMBEDDINGS_LENGTH)