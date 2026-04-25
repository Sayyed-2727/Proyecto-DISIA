import pandas as pd
from sklearn.datasets import fetch_california_housing
from pathlib import Path
import argparse
import sys

def ingest_data(output_path: str):
    """
    Simula la ingesta de datos desde una fuente externa y los guarda en crudo.
    """
    print("Iniciando la conexión a la fuente de datos...")

    try:
        # 1. Extracción: Obtenemos los datos (simulando una query o petición API)
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        print("Datos extraídos correctamente.")

        # 2. Gestión de rutas: Aseguramos que la carpeta de destino existe
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 3. Almacenamiento: Guardamos la foto "raw" inmutable
        df.to_csv(output_file, index=False)
        print(f"Éxito: {len(df)} registros guardados en '{output_file}'")

    except Exception as e:
        print(f"Error crítico durante la ingesta: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Permite ejecutar el script pasando la ruta de destino como argumento
    parser = argparse.ArgumentParser(description="Pipeline de Ingesta de Datos")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/california_housing.csv",
        help="Ruta donde se guardará el CSV en crudo"
    )
    args = parser.parse_args()

    ingest_data(args.output)