import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

PROCESSED_DATA_PATH = "/opt/airflow/data/processed/featured_data.csv"


def train_model():
    print("Model eğitimi başlatılıyor...")

    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    mlflow.set_experiment("VitalWatch Anomaly Detection")

    df = pd.read_csv(PROCESSED_DATA_PATH)

    features = ["rolling_mean", "rolling_std"]
    X = df[features]
    y_true = df["is_anomaly"]

    with mlflow.start_run():
        contamination = y_true.mean()
        model = IsolationForest(
            n_estimators=100, contamination=contamination, random_state=42
        )
        model.fit(X)

        y_pred_scores = model.decision_function(X) * -1
        y_pred_labels = [1 if score > 0 else 0 for score in model.predict(X)]

        roc_auc = roc_auc_score(y_true, y_pred_scores)
        pr_auc = average_precision_score(y_true, y_pred_scores)

        print(f"Model Eğitildi. ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="isolation_forest_model",
            registered_model_name="isolation_forest_model",
        )

        print("Model ve metrikler MLflow'a başarıyla kaydedildi.")


if __name__ == "__main__":
    train_model()
