# dags/data_pipeline_dag.py

from __future__ import annotations

import pendulum
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

PROJECT_HOME = "/opt/airflow"
MODEL_NAME = "isolation_forest_model"

with DAG(
    dag_id="vitalwatch_data_pipeline",
    start_date=pendulum.datetime(2025, 10, 17, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=["vitalwatch", "data_pipeline", "training", "canary"],
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

    # Görev 3: Model Eğitim Script'ini Çalıştırma
    task_train_model = BashOperator(
        task_id="train_model",
        bash_command=f"python {PROJECT_HOME}/src/model_training/train.py",
    )

    # Görev 4: Yeni eğitilen modeli "Staging" aşamasına taşı
    task_promote_to_staging = BashOperator(
        task_id="promote_model_to_staging",
        bash_command=(
            f"python {PROJECT_HOME}/src/model_training/promote_model.py "
            f"--model-name {MODEL_NAME} "
            f"--stage Staging"
        ),
    )

    # Görevler arasındaki bağımlılık zinciri
    (
        task_generate_data
        >> task_extract_features
        >> task_train_model
        >> task_promote_to_staging
    )
