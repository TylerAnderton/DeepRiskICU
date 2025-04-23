import pandas as pd
from unittest.mock import patch

from ..normal_features import get_patient_admission_data, calculate_ages, encode_categorical_features
from .pipeline_test_case import MLTestCase

class NormalFeaturesTestCase(MLTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    # @patch('ml.normal_features.joblib.load')
    def test_get_patient_admission_data(self):
        """Test patient and admission data processing"""
        result = get_patient_admission_data(self.sample_input)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], 1)
        self.assertIn('admission_dt', result.columns)
        self.assertIn('dob', result.columns)
        self.assertIn('gender', result.columns)
        
    def test_calculate_ages(self):
        """Test age calculation"""
        patient_data = get_patient_admission_data(self.sample_input)
        result = calculate_ages(patient_data)
        
        self.assertIn('age_yrs', result.columns)
        print(f"Standardized Age: {result['age_yrs'].iloc[0]}")
    # @patch('ml.normal_features.joblib.load')
    def test_encode_categorical_features(self):
        """Test categorical feature encoding"""
        # mock_joblib.return_value = self.mock_encoder
        
        patient_data = get_patient_admission_data(self.sample_input)
        patient_data = calculate_ages(patient_data)
        result = encode_categorical_features(patient_data, "dummy_path")
        
        self.assertIn('gender_male', result.columns)
        self.assertNotIn('gender', result.columns)  # Original column should be dropped
        self.assertNotIn('dob', result.columns)  # Should be dropped