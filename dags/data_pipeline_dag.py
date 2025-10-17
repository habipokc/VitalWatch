# dags/data_pipeline_dag.py

from __future__ import annotations

import pendulum
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

# Konteyner içindeki mutlak yol
PROJECT_HOME = "/opt/airflow"

with DAG(
    dag_id="vitalwatch_data_pipeline",
    start_date=pendulum.datetime(2025, 10, 17, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=["vitalwatch", "data_pipeline", "training"],
) as dag:

    # Görev 1: Veri Simülatörünü Çalıştırma
    task_generate_data = BashOperator(
        task_id="generate_simulated_data",
        bash_command=f"python {PROJECT_HOME}/src/data_pipeline/simulator.py",
    )

    # Görev 2: Özellik Çıkarıcıyı Çalıştırma
    task_extract_features = BashOperator(
        task_id="extract_features",
        bash_command=f"python {PROJECT_HOME}/src/data_pipeline/feature_extractor.py",
    )

    # ==========================================================
    # ===== YENİ GÖREVİ BURAYA EKLİYORUZ =======================
    # ==========================================================
    # Görev 3: Model Eğitim Script'ini Çalıştırma
    task_train_model = BashOperator(
        task_id="train_model",
        bash_command=f"python {PROJECT_HOME}/src/model_training/train.py",
    )
    # ==========================================================
    # ==========================================================

    # Görevler Arasındaki Bağımlılığı Tanımlama
    # Zinciri güncelliyoruz: Veri üret -> Özellik çıkar -> MODELİ EĞİT
    task_generate_data >> task_extract_features >> task_train_model
