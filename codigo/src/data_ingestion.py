import pandas as pd
from pathlib import Path
import argparse
import sys
import shutil

def ingest_data(output_path: str, source_path: str = "/app/Hito 2/csvs/processed_features_48h_setA.csv"):
    """
    Ingesta de datos del hospital MIMIC desde el archivo procesado.
    """
    print("Iniciando la conexión a la fuente de datos del hospital MIMIC...")

    try:
        # 1. Extracción: Cargamos los datos del hospital MIMIC
        if not Path(source_path).exists():
            raise FileNotFoundError(f"No se encuentra el dataset MIMIC en: {source_path}")
        
        df = pd.read_csv(source_path)
        print(f"Datos del hospital MIMIC extraídos correctamente: {df.shape}")

        # 2. Gestión de rutas: Aseguramos que la carpeta de destino existe
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 3. Almacenamiento: Guardamos los datos MIMIC en la ruta de trabajo
        df.to_csv(output_file, index=False)
        print(f"Éxito: {len(df)} registros del hospital MIMIC guardados en '{output_file}'")
        print(f"Columnas disponibles: {list(df.columns)}")
        print(f"Target variable: In-hospital_death")

    except Exception as e:
        print(f"Error crítico durante la ingesta: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Permite ejecutar el script pasando la ruta de destino como argumento
    parser = argparse.ArgumentParser(description="Pipeline de Ingesta de Datos MIMIC")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/mimic_hospital_data.csv",
        help="Ruta donde se guardará el CSV en crudo"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="/app/Hito 2/csvs/processed_features_48h_setA.csv",
        help="Ruta del dataset MIMIC de origen"
    )
    args = parser.parse_args()

    ingest_data(args.output, args.source)
