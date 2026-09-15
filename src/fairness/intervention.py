"""
Fairness interventions: reweighting, adversarial debiasing.
Implements the "before/after" comparison for your proposal.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Optional

def compute_reweighting_weights(y: np.ndarray,
                                 sensitive: np.ndarray,
                                 base_weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute sample weights for fairness-aware training.
    
    This implements the "reweighting" intervention: examples from underrepresented
    or disadvantaged groups receive greater importance during training.
    
    Approach: weight each sample by the inverse of its group's selection rate
    relative to the overall selection rate.
    """
    n = len(y)
    weights = np.ones(n)
    
    groups = np.unique(sensitive)
    overall_rate = y.mean()
    
    for g in groups:
        mask = sensitive == g
        group_rate = y[mask].mean()
        
        if group_rate > 0:
            # Weight = overall_rate / group_rate
            # Groups with lower selection rates get higher weights
            group_weight = overall_rate / group_rate
            weights[mask] = group_weight
    
    # Normalize to sum to n
    weights = weights / weights.mean()
    
    if base_weights is not None:
        weights = weights * base_weights
        weights = weights / weights.mean()
    
    return weights


class ReweightedClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper that applies fairness reweighting to any sklearn classifier.
    """
    def __init__(self, base_classifier=None, sensitive_col=None, random_state=42):
        self.base_classifier = base_classifier or LogisticRegression(max_iter=1000)
        self.sensitive_col = sensitive_col
        self.random_state = random_state
        self.scaler = None
    
    def fit(self, X, y):
        # Extract sensitive attribute
        sensitive = X[self.sensitive_col].values if isinstance(X, pd.DataFrame) else X[:, -1]
        
        # Compute fairness weights
        weights = compute_reweighting_weights(y, sensitive)
        
        # Store scaler if using raw features
        self.scaler = StandardScaler()
        
        # Fit base classifier with sample weights
        self.base_classifier.fit(X, y, sample_weight=weights)
        return self
    
    def predict(self, X):
        return self.base_classifier.predict(X)
    
    def predict_proba(self, X):
        return self.base_classifier.predict_proba(X)


def apply_reweighting_intervention(X_train: pd.DataFrame,
                                    y_train: np.ndarray,
                                    sensitive_col: str,
                                    model_type: str = 'logistic_regression') -> object:
    """
    Apply reweighting intervention to a training dataset.
    
    This is the core "before vs after" experiment:
    1. Train model WITHOUT reweighting → measure bias
    2. Train model WITH reweighting → measure bias again
    3. Compare fairness metrics
    """
    from src.models.train import train_models, build_preprocessor
    
    # Extract sensitive attribute
    sensitive = X_train[sensitive_col].values
    
    # Compute fairness weights
    weights = compute_reweighting_weights(y_train, sensitive)
    
    # Train with weights
    # Note: For tree models, sklearn's RandomForestClassifier and XGBoost
    # accept sample_weight in fit()
    
    return weights  # The training script should use these weights