from django.conf import settings

from ..pipeline import inference_xgb
from .pipeline_test_case import MLTestCase

ml_model_type = settings.ML_MODEL_TYPE

one_hot_encoder_path = settings.ONE_HOT_ENCODER_PATH
xgboost_dir = settings.XGBOOST_DIR
xgboost_model_path = settings.XGBOOST_MODEL_PATH
xgboost_pca_path = settings.XGBOOST_PCA_PATH

neural_dir = settings.NEURAL_DIR
neural_model_path = settings.NEURAL_MODEL_PATH
neural_pca_dict_path = settings.NEURAL_PCA_DICT_PATH
neural_hidden_dim = settings.NEURAL_HIDDEN_DIM
neural_model_type = settings.NEURAL_MODEL_TYPE
neural_dropout = settings.NEURAL_DROPOUT

embedding_cols = settings.EMBEDDING_COLS


class InferenceTestCase(MLTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    # @patch('ml.inference.joblib.load')
    def test_inference_xgb(self):
        """Test XGBoost inference"""
        # Mock PCA and model
        # mock_pca = MagicMock()
        # mock_pca.transform.return_value = np.array([[0.1, 0.2, 0.3]])
        # mock_pca.explained_variance_ratio_ = np.array([0.5, 0.3, 0.2])
        
        # mock_pca_dict = {
        #     'note_embedding': mock_pca,
        #     'prescription_embedding': mock_pca
        # }
        
        # mock_model = MagicMock()
        # mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
        
        # mock_joblib.side_effect = [mock_pca_dict, mock_model]
        
        # # Create test features
        # features = pd.DataFrame({
        #     'age_yrs': [56.0],
        #     'gender_male': [1.0],
        #     'note_embedding': [self.mock_embedding],
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

        merged_features = merge_features(
            patient_admission_features,
            note_features,
            prescription_features
        )
        merged_features = fill_missing_embeddings(merged_features)
        
        if ml_model_type == 'xgboost':
            result = inference_xgb(
                features=merged_features,
                model_path=xgboost_model_path,
                pca_dict_path=xgboost_pca_path,
                embedding_cols=embedding_cols,
            )
            self.assertAlmostEqual(result, 0.1973, delta=0.001)

        if ml_model_type == 'neural':
            result = inference_neural(
                features=merged_features,
                embedding_cols=embedding_cols,
                model_path=neural_model_path,
                pca_dict_path=neural_pca_dict_path,
                hidden_dim=neural_hidden_dim,
                model_type=neural_model_type,
                dropout=neural_dropout,
            )
            self.assertAlmostEqual(result, 0.0633, delta=0.001)