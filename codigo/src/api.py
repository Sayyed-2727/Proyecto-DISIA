import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from infer import MortalityPredictor

# Instanciamos la aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Mortalidad Hospitalaria (MIMIC)",
    description="API MLOps para inferencia en tiempo real - Riesgo de mortalidad en UCI",
    version="2.0.0"
)

# Cargamos el predictor en el contexto global
predictor = MortalityPredictor(
    model_path=os.getenv("MODEL_PATH", "/app/models/best_mortality_model.pkl")
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
def predict_mortality(data: PatientInput):
    """
    Endpoint para predecir riesgo de mortalidad hospitalaria.
    """
    try:
        raw_prediction = predictor.predict(data.dict())

        # En clasificación binaria:
        # asumimos que predict devuelve probabilidad de clase positiva
        mortality_risk = float(raw_prediction)

        return {
            "status": "success",
            "model_version": "2.0",
            "mortality_risk_probability": round(mortality_risk, 4),
            "high_risk": mortality_risk >= 0.5
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error en la predicción: {str(e)}"
        )
