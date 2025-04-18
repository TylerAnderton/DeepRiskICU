import pandas as pd
import numpy as np
import joblib

def apply_inference_PCA_to_embeddings(
    features:pd.DataFrame,
    pca_dict_path,
    embedding_cols=None,
) -> tuple[pd.DataFrame, pd.Series]:
    print('Applying PCA to embeddings...')
    if not embedding_cols:
        embedding_cols = []
        
    standard_cols = [col for col in features.columns if col not in embedding_cols]

    X = features[standard_cols].copy()
    
    pca_objects_dict = joblib.load(pca_dict_path)
        
    for col in embedding_cols:
        embedding_array = np.vstack(features[col].values)
        
        if col not in pca_objects_dict:
            raise ValueError(f"PCA object for {col} not found in pca_objects_dict")
        pca = pca_objects_dict[col]
        reduced_embeddings = pca.transform(embedding_array)

        print(f"Reduced {col} to {reduced_embeddings.shape[1]} components with {pca.explained_variance_ratio_.sum():.4f} variance accounted for")
        
        reduced_embeddings_df = pd.DataFrame(
            reduced_embeddings, 
            columns=[f'{col}_pca_{i}' for i in range(reduced_embeddings.shape[1])]
        )
        assert len(reduced_embeddings_df) == len(features), f"Reduced {col} has different length than features"
        assert reduced_embeddings_df.isna().sum().sum() == 0, f"Reduced {col} has NaN values"
    
        X = pd.concat([X, reduced_embeddings_df], axis=1)
    
    print(f"Final features shape: {X.shape}")    

    return X

def inference_xgb(
    features:pd.DataFrame,
    model_path,
    pca_dict_path,
    embedding_cols=None,
):
    print()
    print('Running inference...')
    X = apply_inference_PCA_to_embeddings(
        features,
        embedding_cols=embedding_cols,
        pca_dict_path=pca_dict_path,
    )
    
    model = joblib.load(model_path)
    probs = model.predict_proba(X)
    
    print(f"XGBoost predicted {probs[0][1]*100:.2f}% mortality risk")
    return probs[0][1]