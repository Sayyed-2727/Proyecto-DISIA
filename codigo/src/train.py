import argparse
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from feature_store import FeatureStore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline


def evaluate_metrics(y_true, y_pred, y_proba):
    auc = roc_auc_score(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred)
    return auc, acc


def main(args):
    print("Iniciando entrenamiento con dataset hospitalario (MIMIC)...")

    input_path = Path(args.input)
    output_model_path = Path(args.output_model)

    # Config MLflow
    mlflow_tracking_uri = args.mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    # Cargar dataset hospitalario procesado
    df = pd.read_csv(input_path)
    print(f"Dataset cargado: {df.shape}")

    # Eliminamos columnas no predictivas si existen
    drop_cols = [col for col in ["subject_id", "hadm_id", "icustay_id", "recordid", "RecordID"] if col in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Variable objetivo (mortalidad hospitalaria)
    target_col = "In-hospital_death"
    if target_col not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{target_col}' en el dataset.")

    # Feature store (imputación profesional para el resto de variables)
    feature_store = FeatureStore.compute_from_training_df(
        df=df,
        target_col=target_col,
        strategy=args.imputation_strategy,
    )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"Características de entrenamiento: {X.shape}")
    print(f"Distribución del target: {y.value_counts().to_dict()}")

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Modelos de clasificación (Challengers)
    models = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {"n_estimators": [200, 400], "max_depth": [6, 10, None]},
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
            "params": {"n_estimators": [200, 400], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
        },
    }

    best_overall_model = None
    best_overall_auc = -1.0
    best_model_name = ""
    best_run_id = ""

    for model_name, config in models.items():
        print(f"\nEntrenando challenger: {model_name}...")

        with mlflow.start_run(run_name=f"train_{model_name}") as run:
            search = RandomizedSearchCV(
                estimator=config["model"],
                param_distributions=config["params"],
                n_iter=args.n_iter,
                scoring="roc_auc",
                cv=args.cv,
                random_state=42,
                n_jobs=-1,
            )

            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]
            auc, acc = evaluate_metrics(y_test, y_pred, y_proba)

            mlflow.log_params(search.best_params_)
            mlflow.log_metric("auc", float(auc))
            mlflow.log_metric("accuracy", float(acc))
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("n_features", int(X.shape[1]))

            # Log feature_store como artifact y también en el bundle del modelo
            fs_path = Path("data/artifacts/feature_store.json")
            feature_store.save(fs_path)
            mlflow.log_artifact(str(fs_path), artifact_path="preprocessing")

            # Log modelo en MLflow SIN "logged models" (evita endpoint /logged-models 404)
            # Usamos artifact_path y log_artifact como alternativa estable.
            model_local_path = Path("data/artifacts/model.pkl")
            model_local_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_model, model_local_path)

            mlflow.log_artifact(str(model_local_path), artifact_path="model")

            print(f"[{model_name}] AUC: {auc:.4f} | Accuracy: {acc:.4f}")
            print("Mejores parámetros:", search.best_params_)

            if auc > best_overall_auc:
                best_overall_auc = auc
                best_overall_model = best_model
                best_model_name = model_name
                best_run_id = run.info.run_id

    # ==============================
    # REGISTRO COMPATIBLE CON TU VERSIÓN DE MLFLOW
    # ==============================
    print(f"\nRegistrando modelo en MLflow Model Registry como: {args.registered_model_name}")

    # 1️⃣ Guardar modelo como artifact estándar (SIN logged-models API)
    model_uri = f"runs:/{best_run_id}/model"

    # 2️⃣ Crear modelo registrado si no existe
    client = mlflow.tracking.MlflowClient()
    try:
        client.create_registered_model(args.registered_model_name)
        print(f"Modelo registrado '{args.registered_model_name}' creado.")
    except Exception:
        print(f"Modelo registrado '{args.registered_model_name}' ya existe.")

    # 3️⃣ Crear nueva versión manualmente (compatible con MLflow antiguo)
    model_version = client.create_model_version(
        name=args.registered_model_name,
        source=model_uri,
        run_id=best_run_id,
    )

    latest_version = model_version.version

    # 4️⃣ Promover automáticamente a Production
    client.transition_model_version_stage(
        name=args.registered_model_name,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True,
    )

    # Guardar bundle local para serving (champion)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "model": best_overall_model,
        "feature_store": feature_store.to_json_dict(),
        "metrics": {"auc": float(best_overall_auc), "model_name": best_model_name},
        "mlflow": {
            "tracking_uri": mlflow_tracking_uri,
            "best_run_id": best_run_id,
            "registered_model": args.registered_model_name,
            "version": latest_version,
        },
    }
    joblib.dump(model_bundle, output_model_path)

    print("\nEntrenamiento finalizado.")
    print(f"Mejor modelo (Champion): {best_model_name} (AUC={best_overall_auc:.4f})")
    print(f"Modelo registrado como '{args.registered_model_name}' versión {latest_version} (Production)")
    print(f"Modelo bundle guardado en {output_model_path}")
    print(f"MLflow tracking URI: {mlflow_tracking_uri}")
    print(f"MLflow best_run_id: {best_run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de modelo de predicción de mortalidad MIMIC")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/mimic_features.csv",
        help="Ruta del dataset procesado con features",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="models/best_mortality_model.pkl",
        help="Ruta donde se guardará el mejor modelo (bundle)",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="http://mlflow:5000",
        help="MLflow Tracking URI (desde contenedor docker-compose)",
    )
    parser.add_argument(
        "--mlflow_experiment",
        type=str,
        default="mimic_mortality",
        help="Nombre del experimento en MLflow",
    )
    parser.add_argument(
        "--registered_model_name",
        type=str,
        default="mimic_mortality_model",
        help="Nombre del modelo en MLflow Model Registry",
    )
    parser.add_argument("--n_iter", type=int, default=6, help="Iteraciones RandomizedSearchCV")
    parser.add_argument("--cv", type=int, default=3, help="Folds CV para RandomizedSearchCV")
    parser.add_argument(
        "--imputation_strategy",
        type=str,
        default="median",
        choices=["median", "mean"],
        help="Estrategia de imputación para features no enviadas (profesional).",
    )

    args = parser.parse_args()
    main(args)
