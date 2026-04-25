import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from infer import HousingPredictor

# Instanciamos la aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Precios (California Housing)",
    description="API MLOps para inferencia en tiempo real",
    version="1.0.0"
)

# Cargamos el predictor en el contexto global (se ejecuta al arrancar el servidor)
predictor = HousingPredictor(model_path=os.getenv("MODEL_PATH", "../models/best_model.pkl"))

# Definimos el contrato de entrada (Esquema de datos)
class HousingInput(BaseModel):
    MedInc: float = Field(..., description="Ingreso medio en decenas de miles", json_schema_extra={"example": 8.3252})
    HouseAge: float = Field(..., description="Edad de la casa en años", json_schema_extra={"example": 41.0})
    AveRooms: float = Field(..., description="Promedio de habitaciones", json_schema_extra={"example": 6.9841})
    AveBedrms: float = Field(..., description="Promedio de dormitorios", json_schema_extra={"example": 1.0238})
    Population: float = Field(..., description="Población del bloque", json_schema_extra={"example": 322.0})
    AveOccup: float = Field(..., description="Ocupación promedio", json_schema_extra={"example": 2.5555})
    Latitude: float = Field(..., description="Latitud", json_schema_extra={"example": 37.88})
    Longitude: float = Field(..., description="Longitud", json_schema_extra={"example": -122.23})

@app.post("/predict")
def predict_price(data: HousingInput):
    """
    Endpoint principal para predecir el precio.
    """
    try:
        # data.dict() convierte el objeto Pydantic a un diccionario normal
        raw_prediction = predictor.predict(data.dict())

        # Multiplicamos por 100,000 por la naturaleza del dataset original
        final_price = raw_prediction * 100000

        return {
            "status": "success",
            "model_version": "1.0",
            "estimated_price_usd": round(final_price, 2)
        }
    except Exception as e:
        # Si algo falla en el modelo, devolvemos un error HTTP 400 limpio
        raise HTTPException(status_code=400, detail=f"Error en la predicción: {str(e)}")