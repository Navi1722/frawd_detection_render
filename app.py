# train.py
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np

# ================== Step 0: MLflow Tracking ==================
# For Render, use a file-based SQLite DB inside your project
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("FraudDetection")
client = MlflowClient()
MODEL_NAME = "FraudDetectionModel"

# ================== Step 1: Prepare Dataset ==================
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================== Step 2: Define Models ==================
models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

best_acc = 0
best_model_uri = None

# ================== Step 3: Train and Log Models ==================
for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log parameters & metrics
        mlflow.log_param("model_type", name)
        for k, v in model.get_params().items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                continue

        mlflow.log_metric("accuracy", acc)

        # Log & register model (Render-safe)
        mlflow.sklearn.log_model(
            sk_model=model,
            name=f"{name}_model",          # <- use 'name', not 'artifact_path'
            registered_model_name=MODEL_NAME
        )

        print(f"{name} logged with accuracy: {acc:.4f}")

        # Track best model
        if acc > best_acc:
            best_acc = acc
            best_model_uri = f"models:/{MODEL_NAME}/Production"

# ================== Step 4: Promote Best Model to Production ==================
# Fetch all versions
all_versions = client.get_latest_versions(MODEL_NAME)
version_acc = {}
for v in all_versions:
    run_metrics = client.get_run(v.run_id).data.metrics
    version_acc[int(v.version)] = run_metrics.get("accuracy", 0)

# Determine best
best_version = max(version_acc, key=version_acc.get)

for v in all_versions:
    if int(v.version) == best_version:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=v.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Version {v.version} promoted to Production")
    else:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=v.version,
            stage="Staging"
        )

print(f"Best model URI: models:/{MODEL_NAME}/Production with accuracy {version_acc[best_version]:.4f}")
