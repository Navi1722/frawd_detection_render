# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

app = FastAPI(title="Fraud Detection API")

class InputData(BaseModel):
    features: list[float]  # 20 features

# MLflow setup
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("FraudDetection")
MODEL_NAME = "FraudDetectionModel"
client = MlflowClient()

# Train and register models
def train_and_register_models():
    try:
        client.get_registered_model(MODEL_NAME)
        model_exists = True
    except mlflow.exceptions.RestException:
        model_exists = False

    if model_exists:
        print(f"Model '{MODEL_NAME}' already exists. Skipping training.")
        return f"models:/{MODEL_NAME}/Production"

    # Generate synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                               n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    best_acc = 0
    best_model_name = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            mlflow.log_param("model_type", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, name=f"{name}_model", registered_model_name=MODEL_NAME)

            print(f"{name} logged with accuracy: {acc:.4f}")

            if acc >= best_acc:
                best_acc = acc
                best_model_name = name

    # Promote best model to Production
    all_versions = client.get_latest_versions(MODEL_NAME)
    best_version = max(all_versions, key=lambda v: client.get_run(v.run_id).data.metrics.get("accuracy", 0)).version
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=best_version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Version {best_version} promoted to Production with accuracy {best_acc:.4f}")
    return f"models:/{MODEL_NAME}/Production"

# Load model at startup
MODEL_URI = train_and_register_models()
prod_model = mlflow.pyfunc.load_model(MODEL_URI)

@app.post("/predict")
def predict(input_data: InputData):
    features_array = np.array([input_data.features])
    preds = prod_model.predict(features_array)
    return {"prediction": int(preds[0])}

@app.get("/health")
def health():
    return {"status": "running"}
