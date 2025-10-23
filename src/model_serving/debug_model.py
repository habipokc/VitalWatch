# Bu scripti çalıştırarak MLflow'daki modelleri görebilirsin
# docker-compose exec bentoml python debug_model.py

import os

import mlflow

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

print("📊 MLflow'daki kayıtlı modeller:")
client = mlflow.MlflowClient()

# Tüm kayıtlı modelleri listele
for rm in client.search_registered_models():
    print(f"\n🏷️  Model: {rm.name}")
    for version in rm.latest_versions:
        print(f"   Version {version.version}: Stage={version.current_stage}")
