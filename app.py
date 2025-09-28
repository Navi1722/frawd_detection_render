# app.py
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
import os

# ================== Step 0: Set MLflow tracking URI ==================
# Use local folder for simplicity on Render
mlruns_path = os.path.join(os.getcwd(), "mlruns")
os.makedirs(mlruns_path, exist_ok=True)
mlflow.set_tracking_uri(f"file://{mlruns_path}")

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

mlflow.set_experiment("FraudDetection")
model_name = "FraudDetectionModel"

# Track best model
best_acc = 0
best_model_path = None

# ================== Step 3: Train, Log ==================
for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log model type & hyperparameters
        mlflow.log_param("model_type", name)
        for param_name, param_value in model.get_params().items():
            try:
                mlflow.log_param(param_name, param_value)
            except Exception:
                continue

        # Log accuracy
        mlflow.log_metric("accuracy", acc)

        # Log model
        model_path = f"models/{name}"
        mlflow.sklearn.log_model(model, artifact_path=model_path)

        # Keep track of best model
        if acc > best_acc:
            best_acc = acc
            best_model_path = mlflow.get_artifact_uri(model_path)

        print(f"{name} logged with accuracy: {acc:.4f}")

print(f"Best model path: {best_model_path}, accuracy: {best_acc:.4f}")

# ================== Step 4: Load Best Model ==================
prod_model = mlflow.pyfunc.load_model(best_model_path)

# ================== Step 5: Create FastAPI App ==================
app = FastAPI(title="Fraud Detection API")

class InputData(BaseModel):
    features: list  # list of 20 numeric features

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

