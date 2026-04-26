"""
Simplified training script for MIMIC mortality prediction.
Uses only the 8 features that the dashboard provides.
"""
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

from features import prepare_mimic_features, handle_missing_values


def main():
    print("=== SIMPLIFIED MIMIC MORTALITY TRAINING ===")

    # Load data
    data_path = "/app/Hito 2/csvs/processed_features_48h_setA.csv"
    df_raw = pd.read_csv(data_path)

    # Feature engineering
    df_features = prepare_mimic_features(df_raw)
    df_features = handle_missing_values(df_features)

    # Map dashboard feature names to dataset column names
    feature_mapping = {
        "age": "Age_mean",
        "heart_rate_mean": "HR_mean",
        "sysbp_mean": "SysABP_mean",
        "diasbp_mean": "DiasABP_mean",
        "resp_rate_mean": "RespRate_mean",
        "temperature_mean": "Temp_mean",
        "spo2_mean": "SaO2_mean",
        "glucose_mean": "Glucose_mean",
    }

    available_features = [
        col for col in feature_mapping.values()
        if col in df_features.columns
    ]

    X = df_features[available_features].copy()
    y = df_features["In-hospital_death"].copy()

    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)

    model_bundle = {
        "model": model,
        "features": available_features,
        "feature_mapping": feature_mapping,
        "auc": auc,
        "accuracy": acc,
    }

    with open("/app/models/best_mortality_model.pkl", "wb") as f:
        pickle.dump(model_bundle, f)

    print("✅ Simplified model saved with 8 features")


if __name__ == "__main__":
    main()
