# src/model_training/rollback_model.py

import argparse
import os

from mlflow import MlflowClient


def rollback_staging_model(model_name: str):
    """
    Belirtilen modelin "Staging" aşamasındaki versiyonunu bulup "Archived" yapar.
    """
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not mlflow_tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI ortam değişkeni ayarlanmamış!")

    print(f"MLflow'a bağlanılıyor: {mlflow_tracking_uri}")
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)

    try:
        # "Staging" aşamasındaki versiyonları bul
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])

        if not staging_versions:
            print(
                f"Bilgi: Geri çekilecek bir model bulunamadı ('{model_name}' için Staging versiyonu yok)."
            )
            return

        for version in staging_versions:
            print(f"Staging'deki model bulundu: Versiyon {version.version}")
            print(f"Versiyon {version.version}, 'Archived' aşamasına taşınıyor...")
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived",
                archive_existing_versions=False,
            )
            print(f"Versiyon {version.version} başarıyla arşivlendi!")

    except Exception as e:
        print(f"Model geri çekme sırasında hata oluştu: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, required=True, help="MLflow'daki modelin adı"
    )
    args = parser.parse_args()

    rollback_staging_model(model_name=args.model_name)
