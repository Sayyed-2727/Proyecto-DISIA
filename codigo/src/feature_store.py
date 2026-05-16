import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_INPUT_FEATURES: List[str] = [
    "age",
    "heart_rate_mean",
    "sysbp_mean",
    "diasbp_mean",
    "resp_rate_mean",
    "temperature_mean",
    "spo2_mean",
    "glucose_mean",
]


DEFAULT_FEATURE_MAPPING: Dict[str, str] = {
    "age": "Age_mean",
    "heart_rate_mean": "HR_mean",
    "sysbp_mean": "SysABP_mean",
    "diasbp_mean": "DiasABP_mean",
    "resp_rate_mean": "RespRate_mean",
    "temperature_mean": "Temp_mean",
    "spo2_mean": "SaO2_mean",
    "glucose_mean": "Glucose_mean",
}


@dataclass
class FeatureStore:
    """
    Feature store mínimo para garantizar consistencia train/inference.

    - Mantiene la lista completa de features del dataset (338 aprox)
    - Guarda estadísticas de imputación (mean/median) por columna
    - Expone un método para construir el vector final del modelo
      a partir de sólo 8 inputs del dashboard.

    Esto evita el enfoque no profesional de "poner el resto a 0".
    """

    model_features: List[str]  # columnas usadas para entrenar el modelo
    imputations: Dict[str, float]  # valor por defecto por columna (media/mediana)
    feature_mapping: Dict[str, str]  # dashboard_name -> dataset_feature_name
    input_features: List[str]  # las 8 variables que enviará el dashboard

    @staticmethod
    def compute_from_training_df(
        df: pd.DataFrame,
        target_col: str,
        feature_mapping: Optional[Dict[str, str]] = None,
        input_features: Optional[List[str]] = None,
        strategy: str = "median",
    ) -> "FeatureStore":
        feature_mapping = feature_mapping or DEFAULT_FEATURE_MAPPING
        input_features = input_features or DEFAULT_INPUT_FEATURES

        feature_cols = [c for c in df.columns if c != target_col]

        imputations: Dict[str, float] = {}
        for c in feature_cols:
            s = df[c]
            if not pd.api.types.is_numeric_dtype(s):
                # fallback razonable para no-numéricas: moda o string vacío
                try:
                    val = s.mode(dropna=True).iloc[0]
                except Exception:
                    val = ""
                imputations[c] = val
                continue

            if strategy == "mean":
                val = float(s.mean())
            else:
                val = float(s.median())
            if np.isnan(val):
                val = 0.0
            imputations[c] = val

        return FeatureStore(
            model_features=feature_cols,
            imputations=imputations,
            feature_mapping=feature_mapping,
            input_features=input_features,
        )

    def to_json_dict(self) -> dict:
        return {
            "model_features": self.model_features,
            "imputations": self.imputations,
            "feature_mapping": self.feature_mapping,
            "input_features": self.input_features,
        }

    @staticmethod
    def from_json_dict(d: dict) -> "FeatureStore":
        return FeatureStore(
            model_features=list(d["model_features"]),
            imputations=dict(d["imputations"]),
            feature_mapping=dict(d.get("feature_mapping", DEFAULT_FEATURE_MAPPING)),
            input_features=list(d.get("input_features", DEFAULT_INPUT_FEATURES)),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(path: str | Path) -> "FeatureStore":
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return FeatureStore.from_json_dict(d)

    def build_model_input_from_dashboard(self, dashboard_input: Dict) -> pd.DataFrame:
        """
        Construye un DataFrame con TODAS las features del modelo en orden,
        imputando las que no están presentes con median/mean precomputado,
        y sobreescribiendo las 8 que vienen del dashboard.
        """
        full = dict(self.imputations)

        # Mapear 8 inputs a nombres de dataset
        for dashboard_name in self.input_features:
            if dashboard_name not in dashboard_input:
                raise ValueError(f"Falta input requerido: '{dashboard_name}'")

            dataset_feature = self.feature_mapping.get(dashboard_name, dashboard_name)
            full[dataset_feature] = dashboard_input[dashboard_name]

        # Asegurar columnas en el orden exacto del entrenamiento
        row = {c: full.get(c, 0.0) for c in self.model_features}
        return pd.DataFrame([row], columns=self.model_features)
