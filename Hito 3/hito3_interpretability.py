from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hito3_experiments import (
    BASELINE_HITO2_FEATURES,
    load_or_build_dataset,
    resolve_paths,
    split_dataset,
)


def run_interpretability(project_root: Path, random_state: int = 42) -> dict:
    paths = resolve_paths(project_root)
    tables_dir = paths["hito3_root"] / "artifacts" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    data, data_source = load_or_build_dataset(paths, force_rebuild=False)
    for col in data.columns:
        if col not in ["RecordID", "In-hospital_death"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    train_df, valid_df, test_df = split_dataset(data, random_state=random_state)
    train_valid = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)

    full_feature_cols = [c for c in train_valid.columns if c not in ["RecordID", "In-hospital_death"]]
    y_train_valid = train_valid["In-hospital_death"].astype(int).to_numpy()

    baseline_features = [f for f in BASELINE_HITO2_FEATURES if f in train_valid.columns]

    # Logistic (mejor configuración de n_iter=30)
    X_log = train_valid[baseline_features].copy()
    imputer_log = SimpleImputer(strategy="median")
    scaler_log = StandardScaler()
    X_log_imp = imputer_log.fit_transform(X_log)
    X_log_std = scaler_log.fit_transform(X_log_imp)

    log_model = LogisticRegression(
        solver="liblinear",
        penalty="l1",
        C=0.1,
        class_weight=None,
        max_iter=3000,
        random_state=random_state,
    )
    log_model.fit(X_log_std, y_train_valid)

    coef_df = pd.DataFrame(
        {
            "feature": baseline_features,
            "coef_std": log_model.coef_.ravel(),
            "abs_coef_std": np.abs(log_model.coef_.ravel()),
        }
    ).sort_values("abs_coef_std", ascending=False)
    coef_df.to_csv(tables_dir / "table_7_1_logistic_coefficients.csv", index=False)

    # Random Forest (mejor configuración de n_iter=30)
    rf_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_split=2,
                    min_samples_leaf=5,
                    max_features=0.5,
                    max_depth=16,
                    class_weight=None,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_rf = train_valid[full_feature_cols].copy()
    rf_pipe.fit(X_rf, y_train_valid)

    rf_model = rf_pipe.named_steps["rf"]
    rf_importance = pd.DataFrame(
        {
            "feature": full_feature_cols,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    rf_importance.to_csv(tables_dir / "table_7_2_random_forest_importance.csv", index=False)

    manifest = {
        "dataset_source": data_source,
        "random_state": random_state,
        "n_train_valid": int(len(train_valid)),
        "n_test": int(len(test_df)),
        "baseline_features_used": baseline_features,
        "logistic_top10": coef_df.head(10)[["feature", "coef_std", "abs_coef_std"]].to_dict(orient="records"),
        "rf_top10": rf_importance.head(10).to_dict(orient="records"),
        "xgboost_interpretability": "not_available_in_current_environment",
    }

    with open(paths["hito3_root"] / "artifacts" / "interpretability_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = run_interpretability(root, random_state=42)
    print(json.dumps(result, indent=2, ensure_ascii=False))
