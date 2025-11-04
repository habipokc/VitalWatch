#  src/model_training/promote_model.py

import argparse
import os

from mlflow import MlflowClient


def promote_latest_model(model_name: str, target_stage: str):
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not mlflow_tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI ortam değişkeni ayarlanmamış!")

    print(f"MLflow'a bağlanılıyor: {mlflow_tracking_uri}")
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)

    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        if not all_versions:
            print(f"HATA: '{model_name}' için hiç model versiyonu bulunamadı.")
            return

        latest_version = max(all_versions, key=lambda v: int(v.version))
        print(f"En son versiyon bulundu: Version {latest_version.version}")

        print(
            f"Version {latest_version.version}, '{target_stage}' aşamasına taşınıyor..."
        )
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage=target_stage,
            archive_existing_versions=False,
        )
        print("Başarılı!")

    except Exception as e:
        print(f"Model taşıma sırasında hata oluştu: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, required=True, help="MLflow'daki modelin adı"
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        help="Hedef aşama (örn: Staging, Production)",
    )
    args = parser.parse_args()

    promote_latest_model(model_name=args.model_name, target_stage=args.stage)
