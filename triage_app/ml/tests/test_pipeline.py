# import numpy as np
# from unittest.mock import MagicMock, patch
from django.conf import settings

from ..pipeline import pipe
from .pipeline_test_case import MLTestCase

ml_model_type = settings.ML_MODEL_TYPE


class PipelineTestCase(MLTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    # @patch('ml.normal_features.joblib.load')
    # @patch('ml.inference.joblib.load')
    # @patch('ml.embedded_features.get_bert_embeddings')
    def test_full_pipeline(self):
        """Test the complete pipeline end-to-end"""
        # Mock encoder
        # mock_encoder_load.return_value = self.mock_encoder
        
        # Mock embeddings
        # mock_embeddings.return_value = np.array([self.mock_embedding])
        
        # Mock PCA and model for inference
        # mock_pca = MagicMock()
        # mock_pca.transform.return_value = np.array([[0.1, 0.2, 0.3]])
        # mock_pca.explained_variance_ratio_ = np.array([0.5, 0.3, 0.2])
        
        # mock_pca_dict = {
        #     'note_embedding': mock_pca,
        #     'prescription_embedding': mock_pca
        # }
        
        # mock_model = MagicMock()
        # mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
        
        # mock_inference_load.side_effect = [mock_pca_dict, mock_model]
        
        # Run the full pipeline
        result = pipe(self.sample_input)
        
        # self.assertIsInstance(result, float)
        # self.assertEqual(result, 0.3)

        if ml_model_type == 'xgboost':
            self.assertAlmostEqual(result, 0.0043, delta=0.001)
        elif ml_model_type == 'neural':
            self.assertAlmostEqual(result, 0.0513, delta=0.001)
        
        
    # @patch('ml.normal_features.joblib.load')
    # @patch('ml.inference.joblib.load')
    # @patch('ml.embedded_features.get_bert_embeddings')
    def test_pipeline_no_notes(self):
        """Test pipeline with no clinical notes"""
        # Prepare input with no notes
        input_no_notes = self.sample_input.copy()
        input_no_notes['notes_input'] = []
        
        # Set up mocks
        # mock_encoder_load.return_value = self.mock_encoder
        # mock_embeddings.return_value = np.array([])  # Empty array for no notes
        
        # mock_pca = MagicMock()
        # mock_pca.transform.return_value = np.array([[0.1, 0.2, 0.3]])
        # mock_pca.explained_variance_ratio_ = np.array([0.5, 0.3, 0.2])
        
        # mock_pca_dict = {
        #     'note_embedding': mock_pca,
        #     'prescription_embedding': mock_pca
        # }
        
        # mock_model = MagicMock()
        # mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        
        # mock_inference_load.side_effect = [mock_pca_dict, mock_model]
        
        # This should succeed with zero embeddings filled in
        result = pipe(input_no_notes)
        # self.assertIsInstance(result, float)

        
    # @patch('ml.normal_features.joblib.load')
    # @patch('ml.inference.joblib.load')
    # @patch('ml.embedded_features.get_bert_embeddings')
    def test_pipeline_no_prescriptions(self):
        """Test pipeline with no clinical prescriptions"""
        # Prepare input with no prescriptions
        input_no_prescriptions = self.sample_input.copy()
        input_no_prescriptions['prescriptions_input'] = []
        
        # Set up mocks
        # mock_encoder_load.return_value = self.mock_encoder
        # mock_embeddings.return_value = np.array([])  # Empty array for no prescriptions
        
        # mock_pca = MagicMock()
        # mock_pca.transform.return_value = np.array([[0.1, 0.2, 0.3]])
        # mock_pca.explained_variance_ratio_ = np.array([0.5, 0.3, 0.2])
        
        # mock_pca_dict = {
        #     'note_embedding': mock_pca,
        #     'prescription_embedding': mock_pca
        # }
        
        # mock_model = MagicMock()
        # mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        
        # mock_inference_load.side_effect = [mock_pca_dict, mock_model]
        
        # This should succeed with zero embeddings filled in
        result = pipe(input_no_prescriptions)
        # self.assertIsInstance(result, float)


    # @patch('ml.normal_features.joblib.load')
    # @patch('ml.inference.joblib.load')
    # @patch('ml.embedded_features.get_bert_embeddings')
    def test_pipeline_no_embeddings(self):
        """Test pipeline with no clinical embeddings"""
        # Prepare input with no embeddings
        input_no_embeddings = self.sample_input.copy()
        input_no_embeddings['notes_input'] = []
        input_no_embeddings['prescriptions_input'] = []
        
        # Set up mocks
        # mock_encoder_load.return_value = self.mock_encoder
        # mock_embeddings.return_value = np.array([])  # Empty array for no embeddings
        
        # mock_pca = MagicMock()
        # mock_pca.transform.return_value = np.array([[0.1, 0.2, 0.3]])
        # mock_pca.explained_variance_ratio_ = np.array([0.5, 0.3, 0.2])
        
        # mock_pca_dict = {
        #     'note_embedding': mock_pca,
        #     'prescription_embedding': mock_pca
        # }
        
        # mock_model = MagicMock()
        # mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        
        # mock_inference_load.side_effect = [mock_pca_dict, mock_model]
        
        # This should succeed with zero embeddings filled in
        result = pipe(input_no_embeddings)
        # self.assertIsInstance(result, float)
