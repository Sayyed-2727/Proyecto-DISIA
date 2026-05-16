import pandas as pd
import json
import argparse
from pathlib import Path

def prepare_mimic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara las características del dataset MIMIC para entrenamiento.
    El dataset ya viene con características procesadas del Hito 2.
    """
    df_prepared = df.copy()
    
    # Verificar que existe la columna target
    if 'In-hospital_death' not in df_prepared.columns:
        raise ValueError("Columna target 'In-hospital_death' no encontrada en el dataset")
    
    print(f"Dataset MIMIC preparado con {len(df_prepared.columns)} características")
    print(f"Target: In-hospital_death - Distribución: {df_prepared['In-hospital_death'].value_counts().to_dict()}")
    
    return df_prepared

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maneja valores faltantes en el dataset MIMIC.
    """
    df_clean = df.copy()
    
    # Contar valores faltantes por columna
    missing_counts = df_clean.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"Valores faltantes detectados en {(missing_counts > 0).sum()} columnas")
        # Imputar con la mediana para columnas numéricas
        for col in df_clean.columns:
            if df_clean[col].isnull().any() and pd.api.types.is_numeric_dtype(df_clean[col]):
                median_val = df_clean[col].median()
                df_clean.loc[:, col] = df_clean[col].fillna(median_val)
                print(f"  - {col}: imputado con mediana = {median_val:.2f}")
    
    return df_clean

def save_feature_metadata(df: pd.DataFrame, output_path: Path):
    """
    Guarda metadatos sobre las características para uso en inferencia.
    """
    metadata = {
        "n_features": len(df.columns) - 1,  # Excluir target
        "feature_names": [col for col in df.columns if col != 'In-hospital_death'],
        "target_name": "In-hospital_death",
        "n_samples": len(df),
        "class_distribution": df['In-hospital_death'].value_counts().to_dict()
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    return metadata

# --- BLOQUE DE EJECUCIÓN (Pipeline de preparación de datos) ---

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pipeline de Feature Engineering para MIMIC")
    parser.add_argument("--input", type=str, default="data/raw/mimic_hospital_data.csv", help="Ruta del CSV MIMIC en crudo")
    parser.add_argument("--output", type=str, default="data/processed/mimic_features.csv", help="Ruta destino del CSV procesado")
    parser.add_argument("--metadata", type=str, default="data/artifacts/feature_metadata.json", help="Ruta para guardar metadatos")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    metadata_path = Path(args.metadata)

    print("Iniciando Feature Engineering para datos MIMIC...")

    # 1. Leer datos en crudo
    if not input_path.exists():
        print(f"Error: No se encuentra {input_path}. Ejecuta data_ingestion.py primero.")
        exit(1)

    df = pd.read_csv(input_path)
    print(f"Datos MIMIC leídos: {df.shape}")

    # 2. Preparar características del dataset MIMIC
    df_features = prepare_mimic_features(df)

    # 3. Manejar valores faltantes
    df_clean = handle_missing_values(df_features)
    print(f"Datos después de limpieza: {df_clean.shape}")

    # 4. Guardar metadatos de las características
    metadata = save_feature_metadata(df_clean, metadata_path)
    print(f"Metadatos guardados en {metadata_path}")
    print(f"  - {metadata['n_features']} características")
    print(f"  - {metadata['n_samples']} muestras")

    # 5. Guardar el dataset procesado
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"Dataset MIMIC procesado guardado en {output_path}")
