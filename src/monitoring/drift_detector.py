# src\monitoring\drift_detector.py
import os

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

PROJECT_HOME = "/opt/airflow"
REFERENCE_DATA_PATH = os.path.join(
    PROJECT_HOME, "reports/reference_data/reference_dataset.csv"
)
LIVE_DATA_PATH = os.path.join(PROJECT_HOME, "data/processed/featured_data.csv")
DRIFT_REPORT_PATH = os.path.join(PROJECT_HOME, "reports/data_drift_report.html")


def detect_data_drift():
    print("Veri sapması tespiti başlatılıyor...")
    reference_data = pd.read_csv(REFERENCE_DATA_PATH)
    live_data = pd.read_csv(LIVE_DATA_PATH)
    features = ["rolling_mean", "rolling_std"]
    reference_data_features = reference_data[features]
    live_data_features = live_data[features]

    print("Evidently AI Raporu ve Testleri oluşturuluyor...")

    data_drift_report_and_tests = Report(
        metrics=[DataDriftPreset()], include_tests=True
    )

    report_results = data_drift_report_and_tests.run(
        reference_data=reference_data_features, current_data=live_data_features
    )

    os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)
    report_results.save_html(DRIFT_REPORT_PATH)
    print(f"Veri sapması raporu başarıyla oluşturuldu: {DRIFT_REPORT_PATH}")

    result_dict = report_results.dict()

    # --- NİHAİ DÜZELTME ---
    # 'tests' listesindeki her bir testin durumunu kontrol et.
    # Eğer içlerinden herhangi birinin durumu 'FAIL' ise, drift_detected True olacak.
    drift_detected = any(test["status"] == "FAIL" for test in result_dict["tests"])
    print(f"Sapma tespit edildi mi? -> {drift_detected}")

    if drift_detected:
        print("UYARI: Veri sapması tespit edildi!")
    else:
        print("Veri kalitesi kontrolü başarılı, sapma tespit edilmedi.")


if __name__ == "__main__":
    detect_data_drift()
