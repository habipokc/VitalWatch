#VitalWatch/Dockerfile
FROM apache/airflow:2.8.4-python3.9


COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt