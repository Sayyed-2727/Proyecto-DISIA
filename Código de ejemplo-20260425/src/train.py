import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib

def evaluate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

def main(args):
    print("Iniciando fase de entrenamiento manual...")

    input_path = Path(args.input)
    output_model_path = Path(args.output_model)

    if not input_path.exists():
        print(f"Error: No se encuentra {input_path}")
        return

    df = pd.read_csv(input_path)
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelos a probar (reducido para la clase)
    models = {
        "RandomForest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        },
        "XGBoost": {
            "model": xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
            "params": {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}
        }
    }

    best_overall_model = None
    best_overall_rmse = float('inf')
    best_model_name = ""

    for model_name, config in models.items():
        print(f"\nEntrenando: {model_name}...")
        search = RandomizedSearchCV(
            estimator=config["model"], param_distributions=config["params"],
            n_iter=2, scoring='neg_root_mean_squared_error', cv=3, random_state=42
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        rmse, r2 = evaluate_metrics(y_test, y_pred)

        print(f"[{model_name}] RMSE: {rmse:.4f} | Mejores params: {search.best_params_}")

        # Lógica manual para quedarnos solo con el mejor
        if rmse < best_overall_rmse:
            best_overall_rmse = rmse
            best_overall_model = best_model
            best_model_name = model_name

    # Guardado manual del mejor modelo
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_overall_model, output_model_path)
    print(f"\n¡Entrenamiento finalizado! El mejor fue {best_model_name}.")
    print(f"Modelo guardado en {output_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/california_housing_features.csv")
    parser.add_argument("--output_model", type=str, default="models/best_model.pkl")
    args = parser.parse_args()
    main(args)