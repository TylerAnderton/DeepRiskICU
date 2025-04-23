import joblib
import pandas as pd
import numpy as np
import torch
from torch import nn


class BinaryClassificationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, model_type='small', dropout=0.5):
        super().__init__()
        self.input_dim = input_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        ) if model_type == 'large' else nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*hidden_dim, 1),
        ) if model_type == 'medium' else nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, x):
        assert x.shape[1] == self.input_dim, f"Input should have {self.input_dim} features"
        return self.classifier(x)
    
    
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


def inference_neural(
    features:pd.DataFrame,
    model_path,
    pca_dict_path,
    embedding_cols=None,
    hidden_dim:int=128,
    model_type:str='small',
    dropout:float=0.5,
    device:torch.device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    X = apply_inference_PCA_to_embeddings(
        features,
        embedding_cols=embedding_cols,
        pca_dict_path=pca_dict_path,
    )
    
    model = BinaryClassificationModel(
        input_dim=X.shape[1],
        hidden_dim=hidden_dim,
        model_type=model_type,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        inputs = torch.tensor(X.values, dtype=torch.float32).to(device)
        logits = model(inputs)
        
        probs = torch.sigmoid(logits)
        pred_prob = probs.cpu().numpy()[0][0]
        
    print(f"Neural network predicted {pred_prob*100:.2f}% mortality risk")
    return pred_prob