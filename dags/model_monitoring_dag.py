# dags/model_monitoring_dag.py

from __future__ import annotations

import pendulum
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule  # <- Bu satırı ekliyoruz

PROJECT_HOME = "/opt/airflow"
MODEL_NAME = "isolation_forest_model"  # Model adını bir değişkene alalım

with DAG(
    dag_id="vitalwatch_model_monitoring",
    start_date=pendulum.datetime(2025, 10, 20, tz="UTC"),
    schedule="0 * * * *",  # Her saat başı
    catchup=False,
    tags=["vitalwatch", "monitoring", "drift_detection", "rollback"],
) as dag:

    # Görev 1: Drift tespit etmeye çalış.
    # Bu script, drift varsa `exit 1` ile başarısız olacak.
    task_detect_drift = BashOperator(
        task_id="run_data_drift_detection",
        bash_command=f"python {PROJECT_HOME}/src/monitoring/drift_detector.py",
    )

    # Görev 2: Canary modelini geri çek (Rollback).
    # Bu görev, SADECE bir önceki görev BAŞARISIZ OLDUĞUNDA çalışacak.
    task_trigger_rollback = BashOperator(
        task_id="trigger_automatic_rollback",
        bash_command=(
            f"python {PROJECT_HOME}/src/model_training/rollback_model.py "
            f"--model-name {MODEL_NAME}"
        ),
        trigger_rule=TriggerRule.ONE_FAILED,  # <- Büyü burada!
    )

    # Görevler arasındaki bağımlılığı tanımlıyoruz.
    # Airflow, bu zincirde `task_detect_drift` başarısız olursa,
    # `trigger_rule` sayesinde bir sonraki adımı yine de tetikler.
    task_detect_drift >> task_trigger_rollback
