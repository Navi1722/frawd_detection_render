# app.py
import os
import shutil
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score

# ================== Config ==================
MLRUNS_PATH = os.path.join(os.getcwd(), "mlruns")
os.makedirs(MLRUNS_PATH, exist_ok=True)
mlflow.set_tracking_uri(f"file://{MLRUNS_PATH}")

EXPERIMENT_NAME = "FraudDetection"
mlflow.set_experiment(EXPERIMENT_NAME)

MODEL_NAME = "FraudDetectionModel"
BEST_MODEL_FOLDER = os.path.join(os.getcwd(), "best_model")
os.makedirs(BEST_MODEL_FOLDER, exist_ok=True)
BEST_MODEL_FILE = os.path.join(BEST_MODEL_FOLDER, "best_model_path.txt")

# ================== Dataset ==================
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================== Models ==================
models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# ================== Train & Save Best Model ==================
def train_and_save_best_model():
    best_acc = 0
    best_model_name = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            # Train
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            # Log params and metrics
            mlflow.log_param("model_type", name)
            for param, value in model.get_params().items():
                try:
                    mlflow.log_param(param, value)
                except Exception:
                    continue
            mlflow.log_metric("accuracy", acc)

            # Save model locally if it's the best
            if acc > best_acc:
                best_acc = acc
                best_model_name = name
                best_model_path = os.path.join(BEST_MODEL_FOLDER, f"{name}_model")
                shutil.rmtree(best_model_path, ignore_errors=True)  # clean previous
                mlflow.sklearn.save_model(model, best_model_path)
                with open(BEST_MODEL_FILE, "w") as f:
                    f.write(best_model_path)

            print(f"{name} logged with accuracy: {acc:.4f}")

    print(f"Best model: {best_model_name} with accuracy {best_acc:.4f}")
    return best_model_path

# ================== Load or Train Model ==================
if os.path.exists(BEST_MODEL_FILE):
    with open(BEST_MODEL_FILE, "r") as f:
        best_model_path = f.read().strip()
    print(f"Loading existing best model from {best_model_path}")
else:
    best_model_path = train_and_save_best_model()

prod_model = mlflow.sklearn.load_model(best_model_path)

# ================== FastAPI App ==================
app = FastAPI(title="Fraud Detection API")

class InputData(BaseModel):
    features: list  # expects list of 20 numeric features

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
