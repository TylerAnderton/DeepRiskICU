import numpy as np
import pandas as pd
from .utils import assert_unique

def merge_features(
    patient_admission_features: pd.DataFrame,
    note_features: pd.DataFrame,
    prescription_features: pd.DataFrame
) -> pd.DataFrame:
    print()
    print('Merging features...')
    merged_features = pd.merge(
        patient_admission_features,
        note_features,
        on=['admission_id'],
        how='left'
    )
    assert_unique(merged_features, 'admission_id')

    merged_features = pd.merge(
        merged_features,
        prescription_features,
        on=['admission_id'],
        how='left'
    )
    assert_unique(merged_features, 'admission_id')
    merged_features.drop(columns='admission_id', inplace=True)

    return merged_features

def fill_missing_embeddings(merged_features):
    print('Filling missing embeddings...')
    start_count = len(merged_features)

    if merged_features['note_embedding'].isna().sum() < len(merged_features):
        note_embedding = merged_features[~merged_features['note_embedding'].isna()]['note_embedding'].iloc[0]
        embedding_len = note_embedding.shape[0]
    elif merged_features['prescription_embedding'].isna().sum() < len(merged_features):
        prescription_embedding = merged_features[~merged_features['prescription_embedding'].isna()]['prescription_embedding'].iloc[0]
        embedding_len = prescription_embedding.shape[0]
    else:
        embedding_len = 768 # Default ClinicalBERT embedding length

    null_embedding = np.zeros(embedding_len)
    merged_features['note_embedding'] = merged_features['note_embedding'].apply(
        lambda x: x if isinstance(x, np.ndarray) else null_embedding
    )
    merged_features['prescription_embedding'] = merged_features['prescription_embedding'].apply(
        lambda x: x if isinstance(x, np.ndarray) else null_embedding
    )

    assert merged_features['note_embedding'].apply(lambda x: isinstance(x, np.ndarray)).all(),\
        'note_embedding is not a numpy array'
    assert merged_features['prescription_embedding'].apply(lambda x: isinstance(x, np.ndarray)).all(),\
        'prescription_embedding is not a numpy array'
    assert merged_features['note_embedding'].apply(lambda x: x.shape[0] == embedding_len).all(),\
        "Not all note embeddings have the same length"
    assert merged_features['prescription_embedding'].apply(lambda x: x.shape[0] == embedding_len).all(),\
        "Not all prescription embeddings have the same length"

    assert len(merged_features) == start_count, "Merged features count has changed while filling missing embeddings"

    return merged_features

def assert_types(merged_features, embedding_features=None):
    if embedding_features:
        float_cols = [col for col in merged_features.columns if col not in embedding_features]
        for col in embedding_features:
            assert merged_features[col].apply(lambda x: isinstance(x, np.ndarray)).all(), f"{col} is not a numpy array"
    else:
        float_cols = merged_features.columns

    for col in float_cols:
        assert merged_features[col].apply(lambda x: isinstance(x, float)).all(), f"{col} is not a float"
    
    return True