# VitalWatch: End-to-End MLOps for Time-Series Anomaly Detection

![MLOps CI/CD](https://img.shields.io/badge/MLOps-CI/CD-blue.svg) ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white) ![Airflow](https://img.shields.io/badge/Airflow-017CEE?logo=apache-airflow&logoColor=white) ![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white) ![BentoML](https://img.shields.io/badge/BentoML-F86292?logo=bentoml&logoColor=white)

VitalWatch is a comprehensive, end-to-end MLOps project demonstrating a production-grade workflow for time-series anomaly detection. It showcases best practices for building, deploying, and monitoring a machine learning system in a containerized environment.

The project simulates a real-world scenario of monitoring vital signs, where the goal is to detect anomalies in real-time using an `IsolationForest` model.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [1. Run the Training Pipeline](#1-run-the-training-pipeline)
  - [2. Make Predictions](#2-make-predictions)
  - [3. Monitor the System](#3-monitor-the-system)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

This project implements a full MLOps lifecycle, including:

1.  **Automated Data Pipelines:** An Airflow DAG handles data simulation, feature extraction, and model training.
2.  **Experiment Tracking:** MLflow is used to log experiments, version models, and manage their lifecycle (e.g., Staging, Production).
3.  **Canary Deployments:** An Nginx gateway splits traffic between a stable `production` model and a new `canary` model, allowing for safe, gradual rollouts.
4.  **Automated Monitoring & Rollback:** A second Airflow DAG continuously monitors for data drift. If drift is detected, it automatically triggers a rollback of the canary model to ensure system stability.
5.  **Centralized Serving:** BentoML serves the ML models as high-performance, production-ready APIs, with built-in monitoring capabilities.

## Features

- **End-to-End Automation:** Fully automated CI/CD pipeline for ML models using Apache Airflow.
- **Model Versioning & Registry:** Centralized model management with MLflow.
- **Safe Deployment Strategy:** Canary (or A/B) testing setup with Nginx, routing 10% of traffic to the new model.
- **Drift Detection:** Proactive monitoring for data drift using `Evidently AI`.
- **Automatic Rollback:** Self-healing system that archives the canary model if its performance degrades.
- **Containerized Environment:** All services are containerized with Docker and orchestrated via Docker Compose for portability and scalability.
- **Monitoring Dashboards:** Pre-configured Prometheus and Grafana stack for visualizing key service and model metrics.

## Architecture

The architecture is composed of several interconnected services managed by Docker Compose:

1.  **Nginx Gateway:** The single entry point for all prediction requests. It splits traffic between the production and canary services.
2.  **BentoML Services:**
    - `bentoml_production`: Serves the model tagged as "Production" in MLflow.
    - `bentoml_canary`: Serves the model tagged as "Staging" in MLflow.
3.  **MLflow Server:** The central hub for tracking experiments and managing the model registry.
4.  **Airflow Cluster:**
    - `airflow-scheduler`, `airflow-webserver`, `airflow-worker`: Orchestrate the data and monitoring pipelines.
    - `PostgreSQL`: Serves as the metadata database for Airflow.
    - `Redis`: Acts as the message broker for the Airflow CeleryExecutor.
5.  **Monitoring Stack:**
    - `Prometheus`: Scrapes metrics from the BentoML services.
    - `Grafana`: Provides a dashboard for visualizing the collected metrics.

## Technology Stack

- **Orchestration:** Apache Airflow
- **Model Tracking:** MLflow
- **Model Serving:** BentoML
- **Gateway & Load Balancing:** Nginx
- **Monitoring:** Prometheus, Grafana, Evidently AI
- **Data & ML:** Pandas, Scikit-learn
- **Containerization:** Docker, Docker Compose

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your local machine.
- At least 4GB of RAM allocated to Docker.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/habipokc/VitalWatch.git
    cd VitalWatch
    ```

2.  **Set up Airflow User ID:**
    To avoid permission issues with files created by Airflow, create an `.env` file and set your local user ID.
    ```bash
    echo -e "AIRFLOW_UID=$(id -u)" > .env
    ```

3.  **Build and start all services:**
    ```bash
    docker-compose up --build -d
    ```
    This command will build the custom Airflow image and start all services in detached mode. It may take a few minutes to download all the Docker images for the first time.

## Usage

### 1. Run the Training Pipeline

The `vitalwatch_data_pipeline` DAG is responsible for generating data, training a new model, and promoting it to the "Staging" phase.

1.  Navigate to the Airflow UI at **`http://localhost:8080`**.
2.  Log in with the default credentials (`airflow` / `airflow`).
3.  Find the `vitalwatch_data_pipeline` DAG and un-pause it.
4.  Trigger the DAG manually by clicking the "Play" button.

This will execute the training workflow. Once complete, a new model will be registered in MLflow and deployed to the `canary` service.

### 2. Make Predictions

Send a POST request to the Nginx gateway. It will automatically route your request to either the production or canary model based on the traffic split.

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d ''''''{
    "data": [
      [0.5, 0.2],
      [-1.2, 0.8],
      [10.0, 1.5]
    ]
  }''''''
```
 The model will return `-1` for an anomaly and `1` for a normal data point.

### 3. Monitor the System

- **MLflow UI:** `http://localhost:5000`
  - View experiment runs, compare model metrics, and see which models are in "Staging" vs. "Production".
- **Grafana Dashboard:** `http://localhost:3001`
  - Log in with (`admin` / `admin`). You can create dashboards to visualize metrics scraped by Prometheus.
- **Airflow UI:** `http://localhost:8080`
  - Monitor the status of your DAG runs, including the hourly `vitalwatch_model_monitoring` DAG.

## Project Structure

```
VitalWatch/
├── config/
│   ├── nginx.conf         # Nginx configuration for canary routing
│   └── prometheus.yml     # Prometheus scrape configuration
├── dags/
│   ├── data_pipeline_dag.py # Airflow DAG for training and deployment
│   └── model_monitoring_dag.py # Airflow DAG for drift detection and rollback
├── data/                  # Data storage (mounted in containers)
├── reports/               # Drift reports generated by Evidently AI
├── src/
│   ├── data_pipeline/     # Scripts for data simulation and feature extraction
│   ├── model_serving/     # BentoML service definition and Dockerfile
│   ├── model_training/    # Scripts for model training, promotion, and rollback
│   └── monitoring/        # Drift detection script
├── .env                   # Environment variables (e.g., AIRFLOW_UID)
├── docker-compose.yaml    # Main Docker Compose file to orchestrate all services
├── Dockerfile             # Dockerfile to build the custom Airflow image
└── README.md              # This file
```

## License

This project is licensed under the [MIT License](LICENSE).