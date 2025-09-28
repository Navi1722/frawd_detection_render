import os
import shutil
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score

# ================== CONFIG ==================
BEST_MODEL_FOLDER = os.path.join(os.getcwd(), "best_model")
os.makedirs(BEST_MODEL_FOLDER, exist_ok=True)
BEST_MODEL_FILE = os.path.join(BEST_MODEL_FOLDER, "best_model_path.txt")
mlflow.set_tracking_uri("file://" + os.path.join(os.getcwd(), "mlruns"))  # Optional local tracking

# ================== FASTAPI INIT ==================
app = FastAPI(title="Fraud Detection API")

# ================== DATA & MODELS ==================
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

# ================== TRAINING FUNCTION ==================
def train_and_save_best_model():
    best_acc = 0
    best_model_name = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} accuracy: {acc:.4f}")

        # Log metrics to MLflow
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)
            for k, v in model.get_params().items():
                try:
                    mlflow.log_param(k, v)
                except Exception:
                    continue
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, f"{name}_model")  # Optional

        # Save best model locally
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model_path = os.path.join(BEST_MODEL_FOLDER, f"{name}_model")
            shutil.rmtree(best_model_path, ignore_errors=True)
            mlflow.sklearn.save_model(model, best_model_path)
            with open(BEST_MODEL_FILE, "w") as f:
                f.write(best_model_path)

    print(f"Best model: {best_model_name} with accuracy {best_acc:.4f}")
    return best_model_path

# ================== LOAD OR TRAIN BEST MODEL ==================
if os.path.exists(BEST_MODEL_FILE):
    with open(BEST_MODEL_FILE, "r") as f:
        best_model_path = f.read().strip()
    if not os.path.exists(best_model_path):
        best_model_path = train_and_save_best_model()
else:
    best_model_path = train_and_save_best_model()

prod_model = mlflow.sklearn.load_model(best_model_path)

# ================== REQUEST MODEL ==================
class InputData(BaseModel):
    features: list  # length should be 20

@app.post("/predict")
def predict(data: InputData):
    try:
        arr = np.array(data.features).reshape(1, -1)
        if arr.shape[1] != 20:
            raise HTTPException(status_code=400, detail="Input must have 20 features")
        pred = prod_model.predict(arr)
        return {"prediction": int(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================== HEALTH CHECK ==================
@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}
