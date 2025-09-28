# app.py
import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# ================== Step 0: Setup MLflow ==================
mlruns_path = os.path.join(os.getcwd(), "mlruns")
os.makedirs(mlruns_path, exist_ok=True)
mlflow.set_tracking_uri(f"file://{mlruns_path}")
mlflow.set_experiment("FraudDetection")

MODEL_NAME = "FraudDetectionModel"
BEST_MODEL_PATH_FILE = "best_model_path.txt"

# ================== Step 1: Prepare Dataset ==================
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    n_redundant=5, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================== Step 2: Define Models ==================
models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# ================== Step 3: Train or Load Best Model ==================
def train_and_log_models():
    best_acc = 0
    best_model_uri = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            # Log parameters and metrics
            mlflow.log_param("model_type", name)
            for param_name, param_value in model.get_params().items():
                try:
                    mlflow.log_param(param_name, param_value)
                except Exception:
                    continue
            mlflow.log_metric("accuracy", acc)

            # Log model
            artifact_subpath = f"{name}_model"
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_subpath,
                registered_model_name=MODEL_NAME
            )

            print(f"{name} logged with accuracy: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_model_uri = mlflow.get_artifact_uri(artifact_subpath)

    # Save the best model URI to file so we can load it next time
    with open(BEST_MODEL_PATH_FILE, "w") as f:
        f.write(best_model_uri)

    print(f"Best model URI: {best_model_uri}, accuracy: {best_acc:.4f}")
    return best_model_uri

# Load best model if exists
if os.path.exists(BEST_MODEL_PATH_FILE):
    with open(BEST_MODEL_PATH_FILE, "r") as f:
        best_model_uri = f.read().strip()
    print(f"Loading existing best model from {best_model_uri}")
else:
    best_model_uri = train_and_log_models()

# Load the model for inference
prod_model = mlflow.pyfunc.load_model(best_model_uri)

# ================== Step 4: FastAPI Setup ==================
app = FastAPI(title="Fraud Detection API")

class InputData(BaseModel):
    features: list  # List of 20 numeric features

@app.get("/")
def home():
    return {"message": "Fraud Detection API running on Render!"}

@app.post("/predict")
def predict(data: InputData):
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = prod_model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}
