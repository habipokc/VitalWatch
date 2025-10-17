# src/data_pipeline/simulator.py

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# --- 1. GİRİŞ: Kütüphaneleri ve Temel Ayarları Tanımlama ---
# NumPy ve Pandas'ı projemize dahil ediyoruz.
# os: İşletim sistemiyle ilgili işlemler (dosya yolu oluşturma gibi) için.
# datetime, timedelta: Zaman damgaları oluşturmak için.


def generate_time_series_data(
    start_time_str: str,
    n_samples: int = 1000,
    frequency_seconds: int = 1,
    noise_level: float = 0.05,
    anomaly_prob: float = 0.02,
    anomaly_magnitude: float = 5.0,
) -> pd.DataFrame:
    """
    Normal ve anormal sinyaller içeren sentetik bir zaman serisi verisi üretir.

    Args:
        start_time_str (str): Başlangıç zamanı ('YYYY-MM-DD HH:MM:SS' formatında).
        n_samples (int): Üretilecek veri noktası sayısı.
        frequency_seconds (int): Veri noktaları arasındaki saniye cinsinden aralık.
        noise_level (float): Sinyale eklenecek rastgele gürültünün seviyesi.
        anomaly_prob (float): Herhangi bir veri noktasının anomali olma olasılığı.
        anomaly_magnitude (float): Anomali sinyalinin normalden ne kadar büyük olacağı.

    Returns:
        pd.DataFrame: 'timestamp', 'signal_value' ve 'is_anomaly' sütunlarını içeren DataFrame.
    """

    # --- 2. ZAMAN EKSENİNİ OLUŞTURMA (PANDAS) ---
    # Analoji: Veri tablomuzun ilk sütununu, yani zaman damgalarını oluşturuyoruz.
    # '2025-10-17 12:00:00' gibi bir başlangıç noktasından itibaren,
    # her 1 saniyede bir, toplam 1000 adet zaman damgası üretiyoruz.
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    timestamps = [
        start_time + timedelta(seconds=i * frequency_seconds) for i in range(n_samples)
    ]

    # --- 3. NORMAL SİNYALİ ÜRETME (NUMPY) ---
    # Analoji: Sağlıklı bir EKG sinyalinin ritmik dalgalanmasını taklit ediyoruz.
    # np.linspace: 0'dan 10*pi'ye kadar 1000 eşit aralıklı sayı üretir (zaman ekseni gibi).
    # np.sin: Bu sayılara sinüs fonksiyonu uygulayarak periyodik bir dalga oluşturur.
    time_vector = np.linspace(0, 10 * np.pi, n_samples)
    normal_signal = np.sin(time_vector)

    # Sinyali daha gerçekçi yapmak için biraz "parazit" veya "gürültü" ekliyoruz.
    noise = np.random.normal(0, noise_level, n_samples)
    signal_with_noise = normal_signal + noise

    # --- 4. ANOMALİLERİ EKLEME (NUMPY) ---
    # Analoji: Sağlıklı ritmin arasına aniden giren beklenmedik olayları (ani yükselme) ekliyoruz.
    anomalies = np.zeros(n_samples, dtype=int)  # Başlangıçta hepsi normal (0).
    final_signal = signal_with_noise.copy()

    for i in range(n_samples):
        # Her bir veri noktası için %2 ihtimalle anomali yaratıyoruz.
        if np.random.rand() < anomaly_prob:
            anomalies[i] = 1  # Bu noktayı anomali olarak etiketle (1).
            # Anomaliyi, normal sinyale ani bir sıçrama olarak ekle.
            final_signal[i] += anomaly_magnitude * np.random.choice([-1, 1])

    # --- 5. VERİYİ BİRLEŞTİRME VE KAYDETME (PANDAS) ---
    # Analoji: Tüm parçaları (zaman, sinyal, etiket) akıllı Excel tablomuzda birleştiriyoruz.
    df = pd.DataFrame(
        {"timestamp": timestamps, "signal_value": final_signal, "is_anomaly": anomalies}
    )

    return df


# --- 6. SCRIPT'İ ÇALIŞTIRILABİLİR HALE GETİRME ---
# Bu özel blok (if __name__ == "__main__":), bu script'i doğrudan
# terminalden `python simulator.py` komutuyla çalıştırdığımızda devreye girer.
# Bu, kodun test edilmesini ve yeniden kullanılmasını kolaylaştırır.
if __name__ == "__main__":
    print("Veri simülasyonu başlatılıyor...")

    # Veriyi `data/raw` klasörüne kaydedeceğiz.
    output_dir = "/opt/airflow/data/raw"

    os.makedirs(output_dir, exist_ok=True)  # Klasör yoksa oluştur.
    output_path = os.path.join(output_dir, "simulated_data.csv")

    # Fonksiyonumuzu çağırarak veriyi üretiyoruz.
    simulated_data = generate_time_series_data(start_time_str="2025-10-17 12:00:00")

    # Üretilen DataFrame'i CSV dosyasına kaydediyoruz. index=False önemlidir.
    simulated_data.to_csv(output_path, index=False)

    print(
        f"Başarılı! {len(simulated_data)} satırlık veri '{output_path}' dosyasına kaydedildi."
    )
    print("\nİlk 5 satır:")
    print(simulated_data.head())
