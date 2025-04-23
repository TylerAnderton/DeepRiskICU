from django.conf import settings

from .normal_features import get_patient_admission_data, calculate_ages, encode_categorical_features
from .embedded_features import get_notes_data, concatenate_notes, get_note_embeddings
from .embedded_features import get_prescriptions_data, concatenate_prescriptions, get_prescription_embeddings
from .merge_features import merge_features, fill_missing_embeddings, assert_types
from .inference import inference_xgb, inference_neural

ml_model_dir = settings.ML_MODEL_DIR
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

embedding_cols = ['note_embedding', 'prescription_embedding']

def pipe(input:dict) -> float:
    patient_id = input['patient_input']['patient_id']
    input['admission_input']['patient_id'] = patient_id

    for note in input['notes_input']:
        note['admission_id'] = input['admission_input']['admission_id']
        
    for prescription in input['prescriptions_input']:
        prescription['admission_id'] = input['admission_input']['admission_id']
        
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
    merged_features = fill_missing_embeddings(merged_features)
    assert_types(merged_features, embedding_cols)

    if ml_model_type == 'xgboost':
        return inference_xgb(
            features=merged_features,
            model_path=xgboost_model_path,
            pca_dict_path=xgboost_pca_path,
            embedding_cols=embedding_cols,
        )

    if ml_model_type == 'neural':
        return inference_neural(
            features=merged_features,
            embedding_cols=embedding_cols,
            model_path=neural_model_path,
            pca_dict_path=neural_pca_dict_path,
            hidden_dim=neural_hidden_dim,
            model_type=neural_model_type,
            dropout=neural_dropout,
        )