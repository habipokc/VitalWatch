# src/model_serving/service.py

import os
import traceback

import bentoml
import numpy as np
from bentoml.io import JSON
from pydantic import BaseModel

# --- 1. IO (Giriş/Çıkış) Modelleri ---
# Bu kısım, API'nin ne tür veri beklediğini ve döndüreceğini tanımlar.


class InputData(BaseModel):
    data: list[list[float]]


class OutputData(BaseModel):
    prediction: list[float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    target_stage: str
    mlflow_uri: str
    message: str


class ReloadResponse(BaseModel):
    status: str
    old_version: str
    new_version: str
    message: str


# --- 2. Global Değişkenler ---
# Bu değişkenler, servis çalıştığı sürece hafızada tutulur.

# MLflow sunucusunun adresi ortam değişkeninden alınır.
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Hangi modelin hangi aşamasını yükleyeceğimizi ortam değişkeninden öğreniriz.
# Bu, production ve canary servislerinin farklı modeller yüklemesini sağlar.
MODEL_TAG_STR = os.environ.get(
    "BENTOML_MODEL_TAG", "models:/isolation_forest_model/Production"
)
MODEL_NAME = MODEL_TAG_STR.split("/")[1]
TARGET_STAGE = MODEL_TAG_STR.split("/")[2]

# Yüklenen model ve versiyonu burada saklanır.
model = None
model_version = "none"

# --- 3. Ana Model Yükleme Fonksiyonu ---
# Bu fonksiyon, servisin kalbidir. MLflow'dan doğru modeli bulur ve yükler.


def load_model_from_mlflow():
    """
    Ortam değişkeninde belirtilen model ve aşamayı MLflow'dan yükler.
    Model bulunamazsa hata vermeden devam eder.
    """
    global model, model_version

    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.MlflowClient()

        print(f"\n{'='*60}")
        print(f"🔍 Model aranıyor: '{MODEL_NAME}', Hedef Aşama: '{TARGET_STAGE}'")
        print(f"📍 MLflow URI: {MLFLOW_URI}")
        print(f"{'='*60}")

        versions = client.get_latest_versions(MODEL_NAME, stages=[TARGET_STAGE])

        if not versions:
            print(f"⚠️  '{TARGET_STAGE}' aşamasında model bulunamadı.")
            model = None
            model_version = "none"
            return

        latest_version = versions[0]
        model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"
        print(f"🟢 '{TARGET_STAGE}' modeli bulundu: Versiyon {latest_version.version}")

        print("📦 Model yükleniyor...")
        model = mlflow.pyfunc.load_model(model_uri)
        model_version = latest_version.version
        print(f"✅ Model başarıyla yüklendi! Versiyon: {model_version}")

    except Exception as e:
        print(f"❌ Model yükleme sırasında detaylı hata oluştu: {e}")
        traceback.print_exc()
        model = None
        model_version = "none"


# --- 4. Servis Başlangıcı ---
# Konteyner ayağa kalktığında bu kodlar bir kez çalışır.

print("\n" + "🚀" * 30)
print(f"🚀 VitalWatch BentoML Servisi Başlatılıyor... (Hedef: {TARGET_STAGE})")
print("🚀" * 30 + "\n")
load_model_from_mlflow()

if model is None:
    print("\n" + "⚠️ " * 30)
    print("⚠️  SERVİS MODEL OLMADAN BAŞLATILDI!")
    print(f"⚠️  '{TARGET_STAGE}' aşamasında yüklenecek model bulunamadı.")
    print("⚠️ " * 30 + "\n")

# --- 5. BentoML Servis ve API Tanımları ---

svc = bentoml.Service("vitalwatch_service")


@svc.api(input=JSON(pydantic_model=InputData), output=JSON(pydantic_model=OutputData))
def predict(input_data: InputData) -> OutputData:
    """Anomali tahmini yapar."""
    global model
    if model is None:
        print("❌ Tahmin yapılamıyor: Model yüklü değil.")
        return OutputData(prediction=[-999.0])

    try:
        arr = np.array(input_data.data)
        preds = model.predict(arr)
        preds_list = preds.tolist()
        print(f"✅ Tahmin başarılı: {len(preds_list)} sonuç üretildi.")
        return OutputData(prediction=preds_list)
    except Exception as e:
        print(f"❌ Tahmin sırasında hata: {e}")
        traceback.print_exc()
        return OutputData(prediction=[-999.0])


@svc.api(input=JSON(), output=JSON(pydantic_model=ReloadResponse))
def reload_model(input_data: dict) -> ReloadResponse:
    """Modeli MLflow'dan yeniden yükler."""
    global model_version
    old_version = str(model_version)

    print("\n" + "🔄" * 30)
    print("🔄 Model Yeniden Yükleniyor...")
    load_model_from_mlflow()

    new_version = str(model_version)
    success = model is not None

    return ReloadResponse(
        status="success" if success else "failed",
        old_version=old_version,
        new_version=new_version,
        message=(
            "Model reloaded successfully"
            if success
            else f"No model found for stage '{TARGET_STAGE}'"
        ),
    )


@svc.api(input=JSON(), output=JSON(pydantic_model=HealthResponse))
def health(input_data: dict) -> HealthResponse:
    """Servisin ve modelin sağlık durumunu kontrol eder."""
    is_healthy = model is not None
    message = "Service is healthy and model is loaded"
    if not is_healthy:
        message = f"Service is running but NO MODEL LOADED for stage '{TARGET_STAGE}'!"

    return HealthResponse(
        status="healthy" if is_healthy else "no_model",
        model_loaded=is_healthy,
        model_version=str(model_version),
        target_stage=TARGET_STAGE,
        mlflow_uri=MLFLOW_URI,
        message=message,
    )
