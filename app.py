# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np
from typing import List

app = FastAPI(title="Fraud Detection API")

# Load the Production model from local MLflow registry
MODEL_NAME = "FraudDetectionModel"
MODEL_STAGE = "Production"

try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class Features(BaseModel):
    features: List[float]

@app.get("/")
def root():
    return {"message": "Fraud Detection API is live!"}

@app.post("/predict")
def predict(data: Features):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        X = np.array(data.features).reshape(1, -1)
        pred = model.predict(X)
        return {"prediction": pred.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {e}")

@app.get("/health")
def health():
    return {"status": "healthy" if model else "model not loaded"}
