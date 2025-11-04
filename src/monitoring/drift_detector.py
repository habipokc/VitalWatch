# src/monitoring/drift_detector.py

import os
import sys

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

PROJECT_HOME = "/opt/airflow"
REFERENCE_DATA_PATH = os.path.join(PROJECT_HOME, "data/processed/featured_data.csv")
LIVE_DATA_PATH = os.path.join(PROJECT_HOME, "data/raw/simulated_data.csv")
DRIFT_REPORT_PATH = os.path.join(PROJECT_HOME, "reports/data_drift_report.html")


from data_pipeline.feature_extractor import extract_features


def detect_data_drift():
    print("Veri sapması tespiti başlatılıyor...")

    reference_data = pd.read_csv(REFERENCE_DATA_PATH)

    live_raw_data = pd.read_csv(LIVE_DATA_PATH)
    live_featured_data = extract_features(live_raw_data)

    print(f"Referans veri satır sayısı: {len(reference_data)}")
    print(f"Canlı (işlenmiş) veri satır sayısı: {len(live_featured_data)}")

    features = ["rolling_mean", "rolling_std"]
    reference_data_features = reference_data[features]
    live_data_features = live_featured_data[features]

    print("Evidently AI Raporu oluşturuluyor...")
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(
        reference_data=reference_data_features, current_data=live_data_features
    )

    os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)
    data_drift_report.save_html(DRIFT_REPORT_PATH)
    print(f"Veri sapması raporu başarıyla oluşturuldu: {DRIFT_REPORT_PATH}")

    report_dict = data_drift_report.as_dict()
    drift_detected = report_dict["metrics"][0]["result"]["data_drift"]["data"][
        "metrics"
    ]["dataset_drift"]

    print(f"Veri sapması tespit edildi mi? -> {drift_detected}")

    if drift_detected:
        print(
            "UYARI: Veri sapması tespit edildi! Airflow'a başarısızlık sinyali gönderiliyor."
        )
        sys.exit(1)
    else:
        print(
            "Veri kalitesi kontrolü başarılı, sapma tespit edilmedi. Başarı sinyali gönderiliyor."
        )
        sys.exit(0)


if __name__ == "__main__":
    detect_data_drift()
