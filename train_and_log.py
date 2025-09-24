import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient

# Use local mlruns folder
mlflow.set_tracking_uri("file:mlruns")
mlflow.set_experiment("FraudDetection")
client = MlflowClient()
model_name = "FraudDetectionModel"

# Prepare dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

best_acc = 0
best_version = None

# Train and log models
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
            except:
                pass
        mlflow.log_metric("accuracy", acc)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model",
                                 registered_model_name=model_name)

        print(f"{name} logged with accuracy {acc:.4f}")

# Determine best model version
all_versions = client.get_latest_versions(model_name, stages=[])
version_acc = {int(v.version): client.get_run(v.run_id).data.metrics["accuracy"] for v in all_versions}
best_version = max(version_acc, key=version_acc.get)

# Promote best to Production
for v in all_versions:
    stage = "Production" if int(v.version) == best_version else "Staging"
    client.transition_model_version_stage(name=model_name, version=v.version, stage=stage)
print(f"Best model version: {best_version} promoted to Production")
