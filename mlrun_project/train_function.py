import mlrun
import os
import joblib
from src.data_ingestion import main as ingest_main
from src.features import main as features_main
from src.train import main as train_main


def train_handler(context: mlrun.MLClientCtx):
    """
    MLRun training handler.
    Ejecuta el pipeline completo y registra el modelo en el artifact store.
    """

    # 1️⃣ Ejecutar pipeline
    ingest_main()
    features_main()
    model = train_main()

    # 2️⃣ Guardar modelo localmente
    model_path = "model.pkl"
    joblib.dump(model, model_path)

    # 3️⃣ Loggear modelo en MLRun
    context.log_model(
        key="mimic-triage-model",
        body=model_path,
        model_file="model.pkl",
        framework="sklearn",
        artifact_path=context.artifact_path,
    )

    print("✅ Modelo entrenado y registrado en MLRun")
