import os
import random
import time

import pandas as pd
import psutil
from fastapi import FastAPI, HTTPException, Request, Response
from infer import MortalityPredictor
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest)
from pydantic import BaseModel, Field

from drift import DriftDetector

# Instanciamos la aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Mortalidad Hospitalaria (MIMIC)",
    description="API MLOps para inferencia en tiempo real - Riesgo de mortalidad en UCI",
    version="2.0.0"
)

# Cargamos el predictor en el contexto global
champion_model = MortalityPredictor(
    model_path=os.getenv("MODEL_PATH", "/app/models/best_mortality_model.pkl")
)

# Intentar cargar challenger si existe
challenger_path = "/app/models/candidate_model.pkl"
challenger_model = None
if os.path.exists(challenger_path):
    challenger_model = MortalityPredictor(model_path=challenger_path)

# -----------------------------
# Métricas Prometheus
# -----------------------------
REQUEST_COUNT = Counter(
    "api_request_count",
    "Total number of API requests",
    ["endpoint", "method"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency of API requests in seconds",
    ["endpoint"]
)

PREDICTION_COUNT = Counter(
    "model_prediction_total",
    "Total number of model predictions"
)

MODEL_VERSION_SERVED = Counter(
    "model_version_served_total",
    "Model version served",
    ["version"]
)

MODEL_ERRORS = Counter(
    "model_errors_total",
    "Total number of model prediction errors"
)

# -----------------------------
# Métricas Infraestructura (CPU / RAM)
# -----------------------------
SYSTEM_CPU_USAGE = Gauge(
    "system_cpu_usage_percent",
    "Current system CPU usage percentage"
)

SYSTEM_MEMORY_USAGE = Gauge(
    "system_memory_usage_percent",
    "Current system memory usage percentage"
)

# -----------------------------
# Data Drift Setup
# -----------------------------
DRIFT_THRESHOLD = 50
recent_requests = []

drift_detector = DriftDetector(
    reference_data_path="/app/Hito 2/csvs/processed_features_48h_setA.csv"
)

# Esquema de entrada basado en variables clínicas reales
class PatientInput(BaseModel):
    age: float = Field(..., example=65, description="Edad del paciente")
    heart_rate_mean: float = Field(..., example=85, description="Frecuencia cardíaca media (bpm)")
    sysbp_mean: float = Field(..., example=110, description="Presión arterial sistólica media (mmHg)")
    diasbp_mean: float = Field(..., example=60, description="Presión arterial diastólica media (mmHg)")
    resp_rate_mean: float = Field(..., example=18, description="Frecuencia respiratoria media")
    temperature_mean: float = Field(..., example=37.2, description="Temperatura media (°C)")
    spo2_mean: float = Field(..., example=96, description="Saturación de oxígeno media (%)")
    glucose_mean: float = Field(..., example=140, description="Glucosa media (mg/dL)")

@app.post("/predict")
def predict_mortality(data: PatientInput, request: Request):
    """
    Endpoint para predecir riesgo de mortalidad hospitalaria.
    """
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="/predict", method="POST").inc()

    # Actualizar métricas CPU y RAM
    SYSTEM_CPU_USAGE.set(psutil.cpu_percent())
    SYSTEM_MEMORY_USAGE.set(psutil.virtual_memory().percent)

    try:
        # A/B routing (20% challenger)
        if challenger_model and random.random() < 0.2:
            raw_prediction = challenger_model.predict(data.dict())
            MODEL_VERSION_SERVED.labels(version="challenger").inc()
            model_version = "challenger"
        else:
            raw_prediction = champion_model.predict(data.dict())
            MODEL_VERSION_SERVED.labels(version="champion").inc()
            model_version = "champion"

        PREDICTION_COUNT.inc()

        # Guardamos datos para drift
        recent_requests.append(data.dict())

        if len(recent_requests) >= DRIFT_THRESHOLD:
            current_df = pd.DataFrame(recent_requests)
            drift_detector.compute_drift(current_df)
            recent_requests.clear()

        # En clasificación binaria:
        # asumimos que predict devuelve probabilidad de clase positiva
        mortality_risk = float(raw_prediction)

        response = {
            "status": "success",
            "model_version": model_version,
            "mortality_risk_probability": round(mortality_risk, 4),
            "high_risk": mortality_risk >= 0.5
        }

        # ==============================
        # AUDITORÍA DE PREDICCIONES (HITO 5)
        # ==============================
        import json
        from datetime import datetime
        from pathlib import Path

        audit_path = Path("logs/prediction_audit.jsonl")
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": model_version,
            "inputs": data.dict(),
            "probability": mortality_risk,
            "high_risk": mortality_risk >= 0.5
        }

        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry) + "\n")

        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

        return response

    except Exception as e:
        MODEL_ERRORS.inc()
        raise HTTPException(
            status_code=400,
            detail=f"Error en la predicción: {str(e)}"
        )

# Endpoint de métricas para Prometheus
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok"}
