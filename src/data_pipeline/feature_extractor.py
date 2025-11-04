# src/data_pipeline/feature_extractor.py

import os

import pandas as pd


def extract_features(df: pd.DataFrame, window_size: int = 10) -> pd.DataFrame:
    """
    Ham zaman serisi verisinden hareketli ortalama ve standart sapma gibi
    özellikler çıkarır.

    Args:
        df (pd.DataFrame): 'signal_value' sütunu içeren ham veri.
        window_size (int): Hareketli hesaplamalar için kullanılacak pencere boyutu.

    Returns:
        pd.DataFrame: Yeni özellik sütunları eklenmiş DataFrame.
    """

    # --- 1. HAREKETLİ ORTALAMA HESAPLAMA ---
    # Analoji: Son 10 saniyedeki 'ortalama' sinyal seviyesini hesaplıyoruz.
    # Bu bize sinyalin genel trendini verir.
    df["rolling_mean"] = df["signal_value"].rolling(window=window_size).mean()

    # --- 2. HAREKETLİ STANDART SAPMA HESAPLAMA ---
    # Analoji: Son 10 saniyedeki sinyalin 'dalgalanma' miktarını hesaplıyoruz.
    # Yüksek bir değer, sinyalin son zamanlarda çok oynak olduğunu gösterir.
    df["rolling_std"] = df["signal_value"].rolling(window=window_size).std()

    # --- 3. VERİYİ TEMİZLEME ---
    # Hareketli hesaplamalar, ilk 'window_size - 1' satır için sonuç üretemez
    # (çünkü yeterli geçmiş veri yoktur). Bu satırlar 'NaN' (Not a Number)
    # değeri alır. Modelimizi eğitirken bu eksik değerli satırları istemeyiz.
    # Bu yüzden onları veri setimizden çıkarıyoruz.
    df_featured = df.dropna()

    return df_featured


# --- 4. SCRIPT'İ ÇALIŞTIRILABİLİR HALE GETİRME ---
# Bu script'i doğrudan çalıştırdığımızda aşağıdaki kodlar çalışır.
if __name__ == "__main__":
    print("Özellik çıkarımı başlatılıyor...")

    # Girdi ve çıktı dosyalarının yollarını belirliyoruz.
    input_path = "/opt/airflow/data/raw/simulated_data.csv"
    output_dir = "/opt/airflow/data/processed"
    os.makedirs(output_dir, exist_ok=True)  # Çıktı klasörü yoksa oluştur.
    output_path = os.path.join(output_dir, "featured_data.csv")

    raw_data_df = pd.read_csv(input_path)

    featured_data_df = extract_features(raw_data_df)

    featured_data_df.to_csv(output_path, index=False)

    print(
        f"Başarılı! {len(featured_data_df)} satırlık işlenmiş veri '{output_path}' dosyasına kaydedildi."
    )
    print("\nİlk 5 satır:")
    print(featured_data_df.head())
