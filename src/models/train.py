"""
Train LR, RF, XGBoost on the Blind (unbiased) labels — the baseline.
Save trained models for later use in the counterfactual experiment.
"""
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
import xgboost as xgb


def get_models(random_state: int = 42) -> dict:
    """Return the three model configurations from the proposal."""
    return {
        'logistic_regression': LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=random_state,
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            class_weight='balanced', random_state=random_state, n_jobs=-1,
        ),
        'xgboost': xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=random_state, eval_metric='logloss',
        ),
    }


def evaluate_classifier(model, X, y) -> dict:
    """Standard classification metrics."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'auc': roc_auc_score(y, y_proba),
    }


def train_all(prepared_path: str = 'data/processed/prepared_data.npz',
              output_dir: str = 'results/models') -> dict:
    """Train all three models on Blind labels and save."""
    data = np.load(prepared_path)
    X_train = data['X_train']
    y_train = data['y_train_blind']
    X_test = data['X_test']
    y_test = data['y_test_blind']

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        train_metrics = evaluate_classifier(model, X_train, y_train)
        test_metrics = evaluate_classifier(model, X_test, y_test)

        results[name] = {'train': train_metrics, 'test': test_metrics}
        joblib.dump(model, Path(output_dir) / f'{name}_blind.pkl')

        print(f"  Train: acc={train_metrics['accuracy']:.3f}, "
              f"F1={train_metrics['f1']:.3f}, AUC={train_metrics['auc']:.3f}")
        print(f"  Test:  acc={test_metrics['accuracy']:.3f}, "
              f"F1={test_metrics['f1']:.3f}, AUC={test_metrics['auc']:.3f}")

    import pandas as pd
    rows = []
    for name, m in results.items():
        rows.append({
            'model': name,
            'train_accuracy': m['train']['accuracy'],
            'train_f1': m['train']['f1'],
            'test_accuracy': m['test']['accuracy'],
            'test_precision': m['test']['precision'],
            'test_recall': m['test']['recall'],
            'test_f1': m['test']['f1'],
            'test_auc': m['test']['auc'],
        })
    df = pd.DataFrame(rows)
    df.to_csv('results/tables/baseline_model_performance.csv', index=False)
    print("\nSaved metrics to results/tables/baseline_model_performance.csv")

    return results


if __name__ == '__main__':
    train_all()