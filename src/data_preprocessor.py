"""
Convert FairCVdb profiles into ML-ready feature matrices.
Handles: binary, categorical (one-hot), continuous (standardize) features.
Preserves the original train/test split from FairCVdb.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


def identify_feature_types(profiles: np.ndarray, sample_n: int = 500) -> dict:
    """
    Identify which columns are binary, categorical, or continuous
    by inspecting the number of unique values in a sample.
    """
    sample = profiles[:sample_n]
    binary_cols, categorical_cols, continuous_cols = [], [], []

    for i in range(profiles.shape[1]):
        n_unique = len(np.unique(sample[:, i]))
        if n_unique == 2:
            binary_cols.append(i)
        elif 3 <= n_unique <= 10:
            categorical_cols.append(i)
        else:
            continuous_cols.append(i)

    return {
        'binary': binary_cols,
        'categorical': categorical_cols,
        'continuous': continuous_cols,
    }


def build_preprocessor(feature_types: dict) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer that:
    - Passes binary features through unchanged
    - One-hot encodes categorical features
    - Standardizes continuous features
    """
    return ColumnTransformer(
        transformers=[
            ('binary', 'passthrough', feature_types['binary']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             feature_types['categorical']),
            ('num', StandardScaler(), feature_types['continuous']),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )


def binarize_labels(labels: np.ndarray, threshold: float = None) -> np.ndarray:
    """
    Convert continuous labels into binary 'selected/not-selected'.
    Default threshold: the mean of the training labels.
    """
    if threshold is None:
        threshold = labels.mean()
    return (labels >= threshold).astype(int)


def load_and_prepare(data_path: str = 'data/raw/FairCVdb.npy',
                     output_dir: str = 'data/processed',
                     binarize: bool = True) -> dict:
    """
    Full pipeline: load FairCVdb, build feature matrices, save to disk.
    
    Returns a dict with all train/test arrays and metadata.
    """
    print("Loading FairCVdb...")
    raw = np.load(data_path, allow_pickle=True).item()

    X_train_raw = raw['Profiles Train']
    X_test_raw = raw['Profiles Test']

    # Identify feature types from training data
    feature_types = identify_feature_types(X_train_raw)
    print(f"Feature types: {len(feature_types['binary'])} binary, "
          f"{len(feature_types['categorical'])} categorical, "
          f"{len(feature_types['continuous'])} continuous")

    # Build and fit preprocessor on train only (no leakage)
    preprocessor = build_preprocessor(feature_types)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    print(f"Transformed shapes: train={X_train.shape}, test={X_test.shape}")

    # Labels — three types
    label_keys = {
        'blind': ('Blind Labels Train', 'Blind Labels Test'),
        'gender': ('Biased Labels Train (Gender)', 'Biased Labels Test (Gender)'),
        'ethnicity': ('Biased Labels Train (Ethnicity)', 'Biased Labels Test (Ethnicity)'),
    }

    out = {
        'X_train': X_train,
        'X_test': X_test,
        'feature_types': feature_types,
        'preprocessor': preprocessor,
        'names_train': raw['Names Train'],
        'names_test': raw['Names Test'],
        'bios_train': raw['Bios Train'],
        'bios_test': raw['Bios Test'],
    }

    for short_name, (train_key, test_key) in label_keys.items():
        y_train_raw = np.asarray(raw[train_key]).squeeze()
        y_test_raw = np.asarray(raw[test_key]).squeeze()
        out[f'y_train_{short_name}_continuous'] = y_train_raw
        out[f'y_test_{short_name}_continuous'] = y_test_raw
        if binarize:
            threshold = y_train_raw.mean()  # fit threshold on train only
            out[f'y_train_{short_name}'] = binarize_labels(y_train_raw, threshold)
            out[f'y_test_{short_name}'] = binarize_labels(y_test_raw, threshold)
            out[f'threshold_{short_name}'] = threshold

    # Save to disk
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        Path(output_dir) / 'prepared_data.npz',
        X_train=out['X_train'],
        X_test=out['X_test'],
        y_train_blind=out['y_train_blind'],
        y_test_blind=out['y_test_blind'],
        y_train_gender=out['y_train_gender'],
        y_test_gender=out['y_test_gender'],
        y_train_ethnicity=out['y_train_ethnicity'],
        y_test_ethnicity=out['y_test_ethnicity'],
        y_train_blind_continuous=out['y_train_blind_continuous'],
        y_test_blind_continuous=out['y_test_blind_continuous'],
        y_train_gender_continuous=out['y_train_gender_continuous'],
        y_test_gender_continuous=out['y_test_gender_continuous'],
        y_train_ethnicity_continuous=out['y_train_ethnicity_continuous'],
        y_test_ethnicity_continuous=out['y_test_ethnicity_continuous'],
    )
    joblib.dump(preprocessor, Path(output_dir) / 'preprocessor.pkl')
    joblib.dump(feature_types, Path(output_dir) / 'feature_types.pkl')

    print(f"Saved prepared data to {output_dir}/")

    # Verify binarization balance
    for short_name in ['blind', 'gender', 'ethnicity']:
        pos_rate = out[f'y_train_{short_name}'].mean()
        print(f"  {short_name} positive rate (train): {pos_rate:.3f}")

    return out


if __name__ == '__main__':
    load_and_prepare()