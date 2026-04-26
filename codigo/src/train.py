import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib


def evaluate_metrics(y_true, y_pred, y_proba):
    auc = roc_auc_score(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred)
    return auc, acc


def main(args):
    print("Iniciando entrenamiento con dataset hospitalario (MIMIC)...")

    input_path = Path(args.input)
    output_model_path = Path(args.output_model)

    # Cargar dataset hospitalario procesado
    df = pd.read_csv(input_path)
    print(f"Dataset cargado: {df.shape}")

    # Eliminamos columnas no predictivas si existen
    drop_cols = [col for col in ["subject_id", "hadm_id", "icustay_id", "recordid"] if col in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Variable objetivo (mortalidad hospitalaria)
    target_col = "In-hospital_death"
    if target_col not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{target_col}' en el dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Características de entrenamiento: {X.shape}")
    print(f"Distribución del target: {y.value_counts().to_dict()}")

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Modelos de clasificación
    models = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None]
            }
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42
            ),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]
            }
        }
    }

    best_overall_model = None
    best_overall_auc = 0
    best_model_name = ""

    for model_name, config in models.items():
        print(f"\nEntrenando: {model_name}...")

        search = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["params"],
            n_iter=3,
            scoring="roc_auc",
            cv=3,
            random_state=42,
            n_jobs=-1
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        auc, acc = evaluate_metrics(y_test, y_pred, y_proba)

        print(f"[{model_name}] AUC: {auc:.4f} | Accuracy: {acc:.4f}")
        print("Mejores parámetros:", search.best_params_)

        if auc > best_overall_auc:
            best_overall_auc = auc
            best_overall_model = best_model
            best_model_name = model_name

    # Guardar mejor modelo
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_overall_model, output_model_path)

    print(f"\nEntrenamiento finalizado.")
    print(f"Mejor modelo: {best_model_name} (AUC={best_overall_auc:.4f})")
    print(f"Modelo guardado en {output_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de modelo de predicción de mortalidad MIMIC")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/mimic_features.csv",
        help="Ruta del dataset procesado con features"
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="models/best_mortality_model.pkl",
        help="Ruta donde se guardará el mejor modelo"
    )

    args = parser.parse_args()
    main(args)
