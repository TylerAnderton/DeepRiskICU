import os
from django.conf import settings

from .normal_features import get_patient_admission_data, calculate_ages, encode_categorical_features
from .embedded_features import get_notes_data, concatenate_notes, get_note_embeddings
from .embedded_features import get_prescriptions_data, concatenate_prescriptions, get_prescription_embeddings
from .merge_features import merge_features, fill_missing_embeddings, assert_types
from .inference import inference_xgb

ML_MODEL_DIR = os.path.join(settings.BASE_DIR, 'ml', 'models')
ONE_HOT_ENCODER_PATH = os.path.join(ML_MODEL_DIR, 'one_hot_encoder1.joblib')
XGBOOST_DIR = os.path.join(ML_MODEL_DIR, 'xgboost')
XGBOOST_MODEL_PATH = os.path.join(XGBOOST_DIR, 'xgboost1.joblib')
XGBOOST_PCA_PATH = os.path.join(XGBOOST_DIR, 'pca_objects1.joblib')

EMBEDDING_COLS = ['note_embedding', 'prescription_embedding']

def pipe(input:dict) -> float:
    patient_id = input['patient_input']['patient_id']
    input['admission_input']['patient_id'] = patient_id

    for note in input['notes_input']:
        note['admission_id'] = input['admission_input']['admission_id']
        
    for prescription in input['prescriptions_input']:
        prescription['admission_id'] = input['admission_input']['admission_id']
        
    patient_admission_data = get_patient_admission_data(input)
    patient_admission_data = calculate_ages(patient_admission_data)
    patient_admission_features = encode_categorical_features(patient_admission_data, ONE_HOT_ENCODER_PATH)

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
    assert_types(merged_features, EMBEDDING_COLS)

    return inference_xgb(
        features=merged_features,
        model_path=XGBOOST_MODEL_PATH,
        pca_dict_path=XGBOOST_PCA_PATH,
        embedding_cols=EMBEDDING_COLS,
    )
