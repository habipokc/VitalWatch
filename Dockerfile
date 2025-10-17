# Resmi Airflow 2.8.4 imajını temel alıyoruz.
FROM apache/airflow:2.8.4

# Projemizin ihtiyaç duyduğu kütüphaneleri requirements.txt dosyasından
# imajın içine kalıcı olarak yüklüyoruz.
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt