# Trains an XGBoost classifier on the prepared splits with hyperparameter tuning
# and MLflow experiment tracking, then publishes the best model to the HF Model Hub.
# Invoked by the `model-training` job in the GitHub Actions pipeline.

import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

DATASET_REPO    = "creativitysupreme/tourism-data"
MODEL_REPO      = "creativitysupreme/tourism-predictor"
EXPERIMENT_NAME = "tourism-package-prediction"
MODEL_FILE      = "best_tourism_model.joblib"

# ----------------------------------------------------------------------
# 1. Load the train/test splits from the HF Dataset Hub.
# ----------------------------------------------------------------------
base = f"hf://datasets/{DATASET_REPO}"
Xtrain = pd.read_csv(f"{base}/Xtrain.csv")
Xtest  = pd.read_csv(f"{base}/Xtest.csv")
ytrain = pd.read_csv(f"{base}/ytrain.csv").squeeze("columns")
ytest  = pd.read_csv(f"{base}/ytest.csv").squeeze("columns")
print(f"Loaded splits  ->  Xtrain {Xtrain.shape}, Xtest {Xtest.shape}")

# ----------------------------------------------------------------------
# 2. Feature lists (mirror prep.py).
# ----------------------------------------------------------------------
numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation",
]

# ----------------------------------------------------------------------
# 3. Preprocessing pipeline.
#    SimpleImputer guards against any NaNs that may appear in future data;
#    StandardScaler is kept for parity with any future linear-model swap.
# ----------------------------------------------------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline,     numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

# ----------------------------------------------------------------------
# 4. XGBoost classifier with class-imbalance handling.
# ----------------------------------------------------------------------
scale_pos_weight = (ytrain == 0).sum() / (ytrain == 1).sum()
print(f"Class imbalance ratio (scale_pos_weight) = {scale_pos_weight:.2f}")

xgb_clf = xgb.XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
)
model_pipeline = Pipeline([("preprocess", preprocessor), ("xgb", xgb_clf)])

# ----------------------------------------------------------------------
# 5. Hyperparameter grid (kept small so CI completes in minutes).
# ----------------------------------------------------------------------
param_grid = {
    "xgb__n_estimators":     [100, 150],
    "xgb__max_depth":        [3, 4],
    "xgb__learning_rate":    [0.05, 0.1],
    "xgb__colsample_bytree": [0.7, 1.0],
}
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",          # F1 on the positive class -> robust to imbalance
    n_jobs=-1,
    refit=True,
    verbose=1,
)

# ----------------------------------------------------------------------
# 6. MLflow experiment tracking - every combo + the best run.
# ----------------------------------------------------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="grid_search"):
    grid_search.fit(Xtrain, ytrain)

    # Log every hyperparameter combination as a nested child run.
    for i, params in enumerate(grid_search.cv_results_["params"]):
        with mlflow.start_run(run_name=f"combo_{i}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_f1", grid_search.cv_results_["mean_test_score"][i])
            mlflow.log_metric("std_cv_f1",  grid_search.cv_results_["std_test_score"][i])

    # Best-model summary on the parent run.
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)

    # Test-set evaluation.
    best_model = grid_search.best_estimator_
    y_pred  = best_model.predict(Xtest)
    y_proba = best_model.predict_proba(Xtest)[:, 1]

    test_metrics = {
        "test_accuracy":  accuracy_score(ytest, y_pred),
        "test_precision": precision_score(ytest, y_pred),
        "test_recall":    recall_score(ytest, y_pred),
        "test_f1":        f1_score(ytest, y_pred),
        "test_roc_auc":   roc_auc_score(ytest, y_proba),
    }
    mlflow.log_metrics(test_metrics)

    # Log the trained sklearn pipeline as an MLflow artifact.
    mlflow.sklearn.log_model(best_model, artifact_path="model")

    # Console summary for CI logs and grader visibility.
    print("\n--- Best params ---")
    print(grid_search.best_params_)
    print("\n--- Test metrics ---")
    for k, v in test_metrics.items():
        print(f"{k:>15s}: {v:.4f}")
    print("\n--- Classification report (test) ---")
    print(classification_report(ytest, y_pred))
    print("\n--- Confusion matrix (test) ---")
    print(confusion_matrix(ytest, y_pred))

    # Top-5 hyperparameter combinations from the grid.
    results = pd.DataFrame(grid_search.cv_results_)
    top5 = results.nlargest(5, "mean_test_score")[["params", "mean_test_score", "std_test_score"]]
    print("\n--- Top 5 hyperparameter combinations ---")
    print(top5.to_string(index=False))

# ----------------------------------------------------------------------
# 7. Save the best pipeline and push it to the HF Model Hub.
# ----------------------------------------------------------------------
joblib.dump(best_model, MODEL_FILE)
print(f"\nSaved best model -> {MODEL_FILE}")

api = HfApi(token=os.getenv("HF_TOKEN"))
try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
    print(f"Model repo '{MODEL_REPO}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model repo '{MODEL_REPO}' not found. Creating it...")
    create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=MODEL_FILE,
    path_in_repo=MODEL_FILE,
    repo_id=MODEL_REPO,
    repo_type="model",
)
print(f"Uploaded {MODEL_FILE} -> {MODEL_REPO}")
