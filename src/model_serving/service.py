# C:\90\VitalWatch\src\model_serving\service.py
from typing import Annotated

import bentoml
import mlflow
import numpy as np
from pydantic import BaseModel


# ----------- IO Model tanımı -----------
class InputData(BaseModel):
    data: list[list[float]]


class OutputData(BaseModel):
    prediction: list[float]


# ----------- Servis Tanımı -----------
@bentoml.service(
    name="vitalwatch_service",
    resources={"cpu": "200m", "memory": "256Mi"},
)
class VitalWatchService:
    def __init__(self):
        print("MLflow üzerinden Production model yükleniyor...")

        # MLflow bağlantısı
        mlflow.set_tracking_uri("http://mlflow:5000")
        model_uri = "models:/isolation_forest_model/Production"

        # Modeli çek
        bento_model = bentoml.mlflow.import_model(
            "isolation_forest_model_production",
            model_uri,
        )

        # Modeli yükle
        self.model = bentoml.mlflow.load_model(bento_model.tag)
        print(f"Model başarıyla yüklendi: {bento_model.tag}")

    # ----------- API endpoint -----------
    @bentoml.api
    def predict(
        self,
        input_data: Annotated[InputData, bentoml.io.JSON()],
    ) -> Annotated[OutputData, bentoml.io.JSON()]:
        print(f"Gelen veri: {input_data.data}")
        arr = np.array(input_data.data)
        preds = self.model.predict(arr)
        preds_list = preds.tolist()
        print(f"Tahmin sonucu: {preds_list}")
        return OutputData(prediction=preds_list)
