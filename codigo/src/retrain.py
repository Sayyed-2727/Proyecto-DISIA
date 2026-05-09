import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from model_metrics import ModelMetrics

REFERENCE_DATA_PATH = "data/processed/mimic_features.csv"
MODEL_PATH = "../models/best_model.pkl"
CANDIDATE_PATH = "../models/candidate_model.pkl"

def retrain():

    print("🔁 Iniciando reentrenamiento...")

    df = pd.read_csv(REFERENCE_DATA_PATH)

    X = df.drop("mortality", axis=1)
    y = df["mortality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    y_pred_probs = model.predict_proba(X_test)[:, 1]

    metrics = ModelMetrics.compute_and_export(y_test.tolist(), y_pred_probs.tolist())

    print("📊 Métricas nuevo modelo:", metrics)

    # Evaluar champion actual si existe
    try:
        with open(MODEL_PATH, "rb") as f:
            champion_model = pickle.load(f)

        champion_probs = champion_model.predict_proba(X_test)[:, 1]
        champion_metrics = ModelMetrics.compute_and_export(
            y_test.tolist(), champion_probs.tolist()
        )

        print("📊 Métricas champion actual:", champion_metrics)

        if metrics["f1"] > champion_metrics["f1"]:
            print("🚀 Nuevo modelo mejora al champion. Promoviendo...")
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model, f)
        else:
            print("ℹ️ Nuevo modelo no supera al champion. Guardado como candidate.")
            with open(CANDIDATE_PATH, "wb") as f:
                pickle.dump(model, f)

    except FileNotFoundError:
        print("⚠️ No hay champion previo. Guardando como nuevo best_model.")
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

    return metrics


if __name__ == "__main__":
    retrain()
