import pandas as pd
import pickle
from pathlib import Path
from typing import Optional

from feature_store import FeatureStore


class MortalityPredictor:
    def __init__(self, model_path: str = "models/best_mortality_model.pkl"):
        """
        Inicializa el predictor cargando el modelo en memoria.

        Soporta:
        - Bundle nuevo: {'model': ..., 'feature_store': ..., ...}
        - Bundle anterior: {'model': ..., 'features': [...], 'feature_mapping': {...}}
        - Legacy: modelo suelto
        """
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {model_file}")

        with open(model_file, "rb") as f:
            loaded_obj = pickle.load(f)

        self.feature_store: Optional[FeatureStore] = None

        # Bundle con feature_store (preferido)
        if isinstance(loaded_obj, dict) and "model" in loaded_obj:
            self.model = loaded_obj["model"]

            fs_dict = loaded_obj.get("feature_store")
            if isinstance(fs_dict, dict) and "model_features" in fs_dict and "imputations" in fs_dict:
                self.feature_store = FeatureStore.from_json_dict(fs_dict)
                print("✓ Modelo cargado (bundle con feature_store)")
                print(f"  - Features modelo: {len(self.feature_store.model_features)}")
                print(f"  - Inputs dashboard: {self.feature_store.input_features}")
            else:
                # Compatibilidad con bundle anterior
                self.features = loaded_obj.get("features", [])
                self.feature_mapping = loaded_obj.get("feature_mapping", {})
                print("✓ Modelo cargado (bundle legacy)")
                print(f"  - Features esperadas: {len(self.features)}")
        else:
            # Legacy: solo el modelo
            self.model = loaded_obj
            self.features = []
            self.feature_mapping = {}
            print("✓ Modelo cargado (modelo legacy sin metadata)")

        self.expected_n_features = getattr(self.model, "n_features_in_", None)

    def predict(self, input_data: dict) -> float:
        """
        Predice probabilidad de mortalidad.

        Comportamiento:
        - Si existe feature_store: construye el vector completo (338) a partir de 8 inputs
          y completa el resto con median/mean del training set.
        - Si no existe feature_store: mantiene comportamiento anterior (zeros / fallback).
        """
        if self.feature_store is not None:
            df = self.feature_store.build_model_input_from_dashboard(input_data)
            X_input = df.values
        else:
            # Compatibilidad (no recomendado)
            if getattr(self, "feature_mapping", None):
                converted_data = {}
                for dashboard_name, dataset_name in self.feature_mapping.items():
                    if dashboard_name in input_data:
                        converted_data[dataset_name] = input_data[dashboard_name]
                    else:
                        raise ValueError(f"Feature requerida '{dashboard_name}' no encontrada en input")

                df = pd.DataFrame([converted_data])
                df = df[self.features]
            else:
                df = pd.DataFrame([input_data])

            if self.expected_n_features is not None:
                if getattr(self, "features", None):
                    full_data = {}
                    for feature in self.features:
                        if feature in df.columns:
                            full_data[feature] = df[feature].values[0]
                        else:
                            full_data[feature] = 0.0
                    df = pd.DataFrame([full_data])
                else:
                    import numpy as np

                    expected = self.expected_n_features
                    X_full = np.zeros((df.shape[0], expected))
                    cols_to_copy = min(df.shape[1], expected)
                    X_full[:, :cols_to_copy] = df.values[:, :cols_to_copy]
                    df = pd.DataFrame(X_full)

            X_input = df.values

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_input)[:, 1]
            return float(proba[0])
        pred = self.model.predict(X_input)
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
