import pandas as pd
import pickle
from pathlib import Path


class MortalityPredictor:
    def __init__(self, model_path: str = "models/best_mortality_model.pkl"):
        """
        Inicializa el predictor cargando el modelo en memoria.
        Soporta tanto modelos legacy (solo modelo) como nuevos (bundle con metadata).
        """
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {model_file}")

        with open(model_file, 'rb') as f:
            loaded_obj = pickle.load(f)
        
        # Detectar si es un bundle o modelo legacy
        if isinstance(loaded_obj, dict) and 'model' in loaded_obj:
            # Nuevo formato: bundle con metadata
            self.model = loaded_obj['model']
            self.features = loaded_obj.get('features', [])
            self.feature_mapping = loaded_obj.get('feature_mapping', {})
            print(f"✓ Modelo cargado (bundle format)")
            print(f"  - Features esperadas: {len(self.features)}")
            print(f"  - Feature mapping: {self.feature_mapping}")
        else:
            # Formato legacy: solo el modelo
            self.model = loaded_obj
            self.features = []
            self.feature_mapping = {}
            print(f"✓ Modelo cargado (legacy format)")
        
        self.expected_n_features = getattr(self.model, "n_features_in_", None)

    def predict(self, input_data: dict) -> float:
        """
        Recibe un diccionario con los datos clínicos y predice
        probabilidad de mortalidad.
        
        Si el modelo tiene feature_mapping, convierte los nombres de features
        del dashboard a los nombres del dataset.
        """
        
        # Si tenemos feature_mapping, convertir nombres
        if self.feature_mapping:
            # Convertir de nombres dashboard a nombres dataset
            converted_data = {}
            for dashboard_name, dataset_name in self.feature_mapping.items():
                if dashboard_name in input_data:
                    converted_data[dataset_name] = input_data[dashboard_name]
                else:
                    raise ValueError(f"Feature requerida '{dashboard_name}' no encontrada en input")
            
            # Crear DataFrame con nombres de columnas del dataset
            df = pd.DataFrame([converted_data])
            
            # Asegurar orden correcto de columnas
            df = df[self.features]
        else:
            # Modo legacy: usar datos tal cual vienen
            df = pd.DataFrame([input_data])
        
        # Validar número de features
        if self.expected_n_features is not None:
            if df.shape[1] != self.expected_n_features:
                raise ValueError(
                    f"El modelo espera {self.expected_n_features} features, "
                    f"pero se recibieron {df.shape[1]}"
                )
        
        # Obtener probabilidad clase positiva (mortalidad = 1)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(df)[:, 1]
            return float(proba[0])
        else:
            # fallback improbable
            pred = self.model.predict(df)
            return float(pred[0])


# Test manual rápido
if __name__ == "__main__":
    predictor = MortalityPredictor()

    datos_prueba = {
        "age": 65,
        "heart_rate_mean": 85,
        "sysbp_mean": 110,
        "diasbp_mean": 60,
        "resp_rate_mean": 18,
        "temperature_mean": 37.2,
        "spo2_mean": 96,
        "glucose_mean": 140,
    }

    riesgo = predictor.predict(datos_prueba)
    print(f"Probabilidad de mortalidad: {riesgo:.4f}")
