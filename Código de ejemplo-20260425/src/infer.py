import pandas as pd
import joblib
from pathlib import Path
from features import create_custom_features

class HousingPredictor:
    def __init__(self, model_path: str = "models/best_model.pkl"):
        """
        Inicializa el predictor cargando el modelo en memoria.
        Hacer esto en el __init__ evita tener que cargar el archivo .pkl
        con cada petición, lo que haría la API lentísima.
        """
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {model_file}")

        self.model = joblib.load(model_file)
        print("Modelo cargado correctamente en memoria.")

    def predict(self, input_data: dict) -> float:
        """
        Recibe un diccionario con los datos raw, aplica las features y predice.
        """
        # 1. Convertir el diccionario a DataFrame (1 sola fila)
        df = pd.DataFrame([input_data])

        # 2. Reutilizar la ingeniería de características (STATELESS)
        df_features = create_custom_features(df)

        # Nota de clase: ¡En inferencia NO quitamos outliers!
        # Si un cliente manda una casa rara, tenemos que darle un precio igual.

        # 3. Predecir
        prediction = self.model.predict(df_features)

        # Devolver el valor numérico puro
        return float(prediction[0])

# Pequeño bloque para probar que el cerebro funciona sin necesidad de API web
if __name__ == "__main__":
    predictor = HousingPredictor()
    datos_prueba = {
        "MedInc": 8.3, "HouseAge": 41.0, "AveRooms": 6.9,
        "AveBedrms": 1.0, "Population": 322.0, "AveOccup": 2.5,
        "Latitude": 37.88, "Longitude": -122.23
    }
    precio = predictor.predict(datos_prueba)
    print(f"Predicción de prueba: {precio * 100000:.2f} USD")