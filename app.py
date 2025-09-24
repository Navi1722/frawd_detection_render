from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np

class Features(BaseModel):
    features: list  # 20 features

app = FastAPI(title="Fraud Detection API")

# Load latest production model from local mlruns
try:
    prod_model_uri = "models:/FraudDetectionModel/Production"
    model = mlflow.pyfunc.load_model(prod_model_uri)
except Exception as e:
    print("Error loading model:", e)
    model = None

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(data: Features):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        features_array = np.array(data.features).reshape(1, -1)
        pred = model.predict(features_array)
        return {"prediction": int(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
