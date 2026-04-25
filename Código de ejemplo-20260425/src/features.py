import pandas as pd
import json
import argparse
from pathlib import Path

def create_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformación SIN estado. Crea nuevas columnas basándose en los datos de la fila.
    Corresponde a la Celda 15 del notebook.
    """
    df_transformed = df.copy()

    # Añadimos una pequeña protección: evitar divisiones por cero en producción
    df_transformed['Households'] = df_transformed.apply(
        lambda row: row['Population'] / row['AveOccup'] if row['AveOccup'] != 0 else 0,
        axis=1
    )

    df_transformed['RoomsPerHousehold'] = df_transformed.apply(
        lambda row: row['AveRooms'] / row['Households'] if row['Households'] != 0 else 0,
        axis=1
    )

    df_transformed['BedroomsToRooms'] = df_transformed.apply(
        lambda row: row['AveBedrms'] / row['AveRooms'] if row['AveRooms'] != 0 else 0,
        axis=1
    )

    return df_transformed

def calculate_iqr_limits(df: pd.DataFrame, target_cols: list = None) -> dict:
    """
    Transformación CON estado. Calcula los límites para eliminar outliers.
    Corresponde a las Celdas 17-18 del notebook.
    Esto SOLO se debe ejecutar durante la fase de entrenamiento.
    """
    if target_cols is None:
        target_cols = df.columns.tolist()

    limits = {}
    for col in target_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            limits[col] = {
                "lower_bound": Q1 - 1.5 * IQR,
                "upper_bound": Q3 + 1.5 * IQR
            }
    return limits

def remove_outliers_from_limits(df: pd.DataFrame, limits: dict) -> pd.DataFrame:
    """
    Aplica los límites calculados previamente para filtrar el dataset.
    Corresponde a la Celda 19 del notebook.
    """
    df_clean = df.copy()
    for col, bounds in limits.items():
        if col in df_clean.columns:
            lb = bounds["lower_bound"]
            ub = bounds["upper_bound"]
            # Nos quedamos solo con las filas que están dentro de los límites
            df_clean = df_clean[(df_clean[col] >= lb) & (df_clean[col] <= ub)]

    return df_clean

# --- BLOQUE DE EJECUCIÓN (Pipeline de preparación de datos) ---

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pipeline de Feature Engineering")
    parser.add_argument("--input", type=str, default="data/raw/california_housing.csv", help="Ruta del CSV en crudo")
    parser.add_argument("--output", type=str, default="data/processed/california_housing_features.csv", help="Ruta destino del CSV procesado")
    parser.add_argument("--limits", type=str, default="data/artifacts/outlier_limits.json", help="Ruta para guardar el artefacto JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    limits_path = Path(args.limits)

    print("Iniciando Feature Engineering...")

    # 1. Leer datos en crudo
    if not input_path.exists():
        print(f"Error: No se encuentra {input_path}. Ejecuta data_ingestion.py primero.")
        exit(1)

    df = pd.read_csv(input_path)
    print(f"Datos leídos: {df.shape}")

    # 2. Crear características (Stateless)
    df_features = create_custom_features(df)

    # 3. Calcular límites de Outliers (Stateful)
    # Nota: Excluimos MedHouseVal para no perder datos si lo usamos de target
    cols_to_check = [c for c in df_features.columns if c != "MedHouseVal"]
    limits = calculate_iqr_limits(df_features, target_cols=cols_to_check)

    # Guardar los límites como un artefacto (CRÍTICO PARA MLOPS)
    limits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(limits_path, "w") as f:
        json.dump(limits, f, indent=4)
    print(f"Límites de outliers guardados en {limits_path}")

    # 4. Aplicar la limpieza de outliers
    df_clean = remove_outliers_from_limits(df_features, limits)
    print(f"Datos después de limpiar outliers: {df_clean.shape} (Eliminadas {df.shape[0] - df_clean.shape[0]} filas)")

    # 5. Guardar el dataset procesado
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"Dataset procesado guardado en {output_path}")