from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np
import pandas as pd

data = np.load('data/processed/prepared_data.npz')
X_train, y_train = data['X_train'], data['y_train_blind']
X_test,  y_test  = data['X_test'],  data['y_test_blind']

# --- Random Forest tuning ---
rf_grid = {
    'max_depth':         [6, 8, 10],
    'min_samples_leaf':  [5, 10, 20],
    'min_samples_split': [10, 20, 50],
}
rf_search = GridSearchCV(
    RandomForestClassifier(n_estimators=300, class_weight='balanced',
                           n_jobs=-1, random_state=42),
    rf_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2,
)
rf_search.fit(X_train, y_train)
print("RF best params:", rf_search.best_params_)
print("RF best CV AUC:", rf_search.best_score_)

rf_best = rf_search.best_estimator_
train_acc = rf_best.score(X_train, y_train)
test_acc  = rf_best.score(X_test, y_test)
print(f"RF tuned — train: {train_acc:.3f}, test: {test_acc:.3f}, "
      f"gap: {train_acc - test_acc:.3f}")

# --- XGBoost tuning ---
xgb_grid = {
    'max_depth':        [4, 6, 8],
    'learning_rate':    [0.05, 0.1, 0.2],
    'subsample':        [0.7, 0.85, 1.0],
    'colsample_bytree': [0.7, 0.85, 1.0],
}
xgb_search = GridSearchCV(
    xgb.XGBClassifier(n_estimators=300, eval_metric='logloss',
                      use_label_encoder=False, random_state=42, n_jobs=-1),
    xgb_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2,
)
xgb_search.fit(X_train, y_train)
print("XGB best params:", xgb_search.best_params_)
print("XGB best CV AUC:", xgb_search.best_score_)

xgb_best = xgb_search.best_estimator_
train_acc = xgb_best.score(X_train, y_train)
test_acc  = xgb_best.score(X_test, y_test)
print(f"XGB tuned — train: {train_acc:.3f}, test: {test_acc:.3f}, "
      f"gap: {train_acc - test_acc:.3f}")

# Save tuned models
import joblib
joblib.dump(rf_best,  'results/models/random_forest_tuned.pkl')
joblib.dump(xgb_best, 'results/models/xgboost_tuned.pkl')