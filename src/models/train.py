"""
Train Logistic Regression, Random Forest, and XGBoost classifiers
for resume screening prediction.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
from pathlib import Path
import yaml

def build_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Build preprocessing pipeline for mixed data types.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='drop'
    )
    return preprocessor


def train_models(X_train: pd.DataFrame, 
                 y_train: pd.Series,
                 numerical_cols: list,
                 categorical_cols: list,
                 random_state: int = 42) -> dict:
    """
    Train all three model types with preprocessing pipeline.
    
    Returns:
        dict mapping model_name -> trained Pipeline
    """
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    
    models = {
        'logistic_regression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                max_iter=1000, 
                random_state=random_state,
                class_weight='balanced'  # Helps with imbalanced selection outcomes
            ))
        ]),
        'random_forest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1
            ))
        ]),
        'xgboost': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state,
                scale_pos_weight=1,  # Adjust based on class balance
                eval_metric='logloss',
                use_label_encoder=False
            ))
        ])
    }
    
    trained = {}
    for name, pipeline in models.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline
        print(f"  ✓ {name} trained")
    
    return trained


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Compute standard ML performance metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'predictions': y_pred,
        'probabilities': y_proba
    }


def save_models(models: dict, output_dir: str):
    """Save trained models to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name, model in models.items():
        joblib.dump(model, output_path / f"{name}.pkl")
        print(f"Saved {name} to {output_path / f'{name}.pkl'}")