from __future__ import annotations

import argparse
import importlib
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

RECALL_25 = "Recall@25"
RECALL_50 = "Recall@50"
RECALL_100 = "Recall@100"
RECALL_200 = "Recall@200"

PURITY_25 = "%fallecidos_top25"
PURITY_50 = "%fallecidos_top50"
PURITY_100 = "%fallecidos_top100"
PURITY_200 = "%fallecidos_top200"

RECALL_COLS = [RECALL_25, RECALL_50, RECALL_100, RECALL_200]
PURITY_COLS = [PURITY_25, PURITY_50, PURITY_100, PURITY_200]


BASELINE_HITO2 = {
    "AUC_ROC": 0.7806,
    "Brier_score": 0.0887,
    RECALL_25: 0.1684,
    RECALL_50: 0.2421,
    RECALL_100: 0.4526,
    RECALL_200: 0.6421,
    PURITY_25: 0.6400,
    PURITY_50: 0.4600,
    PURITY_100: 0.4300,
    PURITY_200: 0.3050,
}

BASELINE_HITO2_FEATURES = [
    "Age_first",
    "Gender_first",
    "ICUType_first",
    "GCS_mean",
    "HR_mean",
    "MAP_mean",
    "SysABP_mean",
    "DiasABP_mean",
    "RespRate_mean",
    "Temp_mean",
    "Creatinine_mean",
    "BUN_mean",
    "WBC_mean",
    "Urine_mean",
]


def resolve_xgb_classifier():
    spec = importlib.util.find_spec("xgboost")
    if spec is None:
        return None
    module = importlib.import_module("xgboost")
    return getattr(module, "XGBClassifier", None)


def parse_time_to_hours(time_series: pd.Series) -> pd.Series:
    parts = time_series.astype(str).str.split(":", expand=True)
    return parts[0].astype(int) + (parts[1].astype(int) / 60.0)


def read_set_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    required_cols = {"Time", "Parameter", "Value"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"{file_path} missing columns: {sorted(missing_cols)}")

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"]).copy()
    df["Hours"] = parse_time_to_hours(df["Time"])
    df["RecordID"] = int(file_path.stem)
    return df[["RecordID", "Time", "Hours", "Parameter", "Value"]]


def build_processed_features(raw_48h: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    raw_48h = raw_48h.sort_values(["RecordID", "Parameter", "Hours"]).copy()
    grouped = raw_48h.groupby(["RecordID", "Parameter"])["Value"]

    features_long = grouped.agg(
        first="first",
        last="last",
        mean="mean",
        min="min",
        max="max",
        std=lambda x: float(np.std(x.to_numpy(), ddof=0)),
        n_mediciones="count",
    ).reset_index()
    features_long["flag_medido"] = 1

    metrics = ["first", "last", "mean", "min", "max", "std", "n_mediciones", "flag_medido"]
    wide_parts = []
    for metric in metrics:
        pivot_metric = features_long.pivot(index="RecordID", columns="Parameter", values=metric)
        pivot_metric.columns = [f"{col}_{metric}" for col in pivot_metric.columns]
        wide_parts.append(pivot_metric)

    feature_matrix = pd.concat(wide_parts, axis=1).reset_index()

    n_cols = [c for c in feature_matrix.columns if c.endswith("_n_mediciones")]
    flag_cols = [c for c in feature_matrix.columns if c.endswith("_flag_medido")]
    for col in n_cols + flag_cols:
        feature_matrix[col] = feature_matrix[col].fillna(0).astype(int)

    processed = outcomes[["RecordID", "In-hospital_death"]].merge(feature_matrix, on="RecordID", how="left")
    processed = processed.sort_values("RecordID").reset_index(drop=True)
    return processed


def resolve_paths(project_root: Path) -> dict[str, Path]:
    hito2_root = project_root / "Hito 2"
    hito3_root = project_root / "Hito 3"
    dataset_root = hito2_root / "predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0"

    return {
        "project_root": project_root,
        "hito2_root": hito2_root,
        "hito3_root": hito3_root,
        "dataset_root": dataset_root,
        "set_a": dataset_root / "set-a",
        "outcomes_a": dataset_root / "Outcomes-a.txt",
        "hito2_csv": hito2_root / "csvs" / "processed_features_48h_setA.csv",
        "hito3_csv": hito3_root / "artifacts" / "processed_features_48h_setA.csv",
        "tables_dir": hito3_root / "artifacts" / "tables",
        "figures_dir": hito3_root / "artifacts" / "figures",
        "predictions_dir": hito3_root / "artifacts" / "predictions",
    }


def ensure_dirs(paths: dict[str, Path]) -> None:
    paths["tables_dir"].mkdir(parents=True, exist_ok=True)
    paths["figures_dir"].mkdir(parents=True, exist_ok=True)
    paths["predictions_dir"].mkdir(parents=True, exist_ok=True)


def load_or_build_dataset(paths: dict[str, Path], force_rebuild: bool = False) -> tuple[pd.DataFrame, str]:
    if paths["hito2_csv"].exists() and not force_rebuild:
        return pd.read_csv(paths["hito2_csv"]), "hito2_csv"

    if paths["hito3_csv"].exists() and not force_rebuild:
        return pd.read_csv(paths["hito3_csv"]), "hito3_csv"

    if not paths["set_a"].exists() or not paths["outcomes_a"].exists():
        raise FileNotFoundError(
            "No se encontró CSV procesado ni fuentes RAW de Set A. Revisa rutas en Hito 2 y dataset PhysioNet 2012."
        )

    set_a_files = sorted(paths["set_a"].glob("*.txt"), key=lambda p: int(p.stem))
    raw_frames = [read_set_file(path) for path in set_a_files]
    raw_all = pd.concat(raw_frames, ignore_index=True)
    raw_48h = raw_all.loc[raw_all["Hours"] <= 48].copy()
    outcomes_a = pd.read_csv(paths["outcomes_a"])

    processed = build_processed_features(raw_48h=raw_48h, outcomes=outcomes_a)
    processed.to_csv(paths["hito3_csv"], index=False)
    return processed, "rebuilt_from_raw"


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logreg_gd(X: np.ndarray, y: np.ndarray, lr: float = 0.05, n_iter: int = 4000, l2: float = 1e-4) -> tuple[np.ndarray, float]:
    n, m = X.shape
    w = np.zeros(m, dtype=float)
    b = 0.0
    for _ in range(n_iter):
        p = sigmoid(X @ w + b)
        err = p - y
        grad_w = (X.T @ err) / n + l2 * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def evaluate_baseline_hito2_replicated(train_valid_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame, list[str]]:
    used_features = [f for f in BASELINE_HITO2_FEATURES if f in train_valid_df.columns]
    if len(used_features) == 0:
        raise ValueError("No hay features baseline de Hito 2 disponibles en el dataset procesado.")

    model_train = train_valid_df[["RecordID", "In-hospital_death", *used_features]].copy()
    model_test = test_df[["RecordID", "In-hospital_death", *used_features]].copy()

    for c in used_features:
        model_train[c] = pd.to_numeric(model_train[c], errors="coerce")
        model_test[c] = pd.to_numeric(model_test[c], errors="coerce")

    medians = model_train[used_features].median()
    X_train = model_train[used_features].fillna(medians).to_numpy(dtype=float)
    X_test = model_test[used_features].fillna(medians).to_numpy(dtype=float)
    y_train = model_train["In-hospital_death"].to_numpy(dtype=float)
    y_test = model_test["In-hospital_death"].to_numpy(dtype=int)

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    w, b = fit_logreg_gd(X_train, y_train)
    y_score = sigmoid(X_test @ w + b)

    metrics = evaluate_predictions(y_true=y_test, y_score=y_score)
    pred_df = pd.DataFrame(
        {
            "RecordID": model_test["RecordID"].to_numpy(),
            "y_true": y_test,
            "y_pred_proba": y_score,
        }
    ).sort_values("y_pred_proba", ascending=False)
    return metrics, pred_df, used_features


def build_feature_reuse_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_feature_cols = [c for c in df.columns if c not in ["RecordID", "In-hospital_death"]]
    all_feature_set = set(all_feature_cols)

    for feature in BASELINE_HITO2_FEATURES:
        rows.append(
            {
                "feature_hito2": feature,
                "presente_en_processed_features_48h_setA": feature in all_feature_set,
                "tipo_esperado_hito2": "estatica" if feature in ["Age_first", "Gender_first", "ICUType_first"] else "agregada_mean_0_48h",
            }
        )

    summary_rows = [
        {
            "feature_hito2": "TOTAL_FEATURES_FULL_PIPELINE",
            "presente_en_processed_features_48h_setA": len(all_feature_cols),
            "tipo_esperado_hito2": "all_columns_except_RecordID_target",
        },
        {
            "feature_hito2": "TOTAL_FEATURES_BASELINE_HITO2_REPLICATED",
            "presente_en_processed_features_48h_setA": sum(r["presente_en_processed_features_48h_setA"] for r in rows),
            "tipo_esperado_hito2": "subset_14_features",
        },
    ]
    return pd.DataFrame(rows + summary_rows)


def ranking_metrics(y_true: np.ndarray, y_score: np.ndarray, ks: tuple[int, ...] = (25, 50, 100, 200)) -> dict[str, float]:
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    total_pos = max(int(y_true.sum()), 1)
    metrics: dict[str, float] = {}

    for k in ks:
        k_eff = min(k, len(y_sorted))
        top_k = y_sorted[:k_eff]
        deaths_top_k = int(top_k.sum())
        metrics[f"Recall@{k}"] = deaths_top_k / total_pos
        metrics[f"%fallecidos_top{k}"] = deaths_top_k / k_eff

    return metrics


def split_dataset(df: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_valid, test = train_test_split(
        df,
        test_size=0.2,
        stratify=df["In-hospital_death"],
        random_state=random_state,
    )

    train, valid = train_test_split(
        train_valid,
        test_size=0.25,
        stratify=train_valid["In-hospital_death"],
        random_state=random_state,
    )

    return train.copy(), valid.copy(), test.copy()


def split_summary(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, part in [("Train", train), ("Validation", valid), ("Test", test)]:
        n = len(part)
        n_pos = int(part["In-hospital_death"].sum())
        rows.append(
            {
                "Split": name,
                "N_estancias": n,
                "N_fallecidos": n_pos,
                "Prevalencia_pct": 100.0 * n_pos / max(n, 1),
            }
        )

    total = pd.concat([train, valid, test], axis=0)
    n_total = len(total)
    n_pos_total = int(total["In-hospital_death"].sum())
    rows.append(
        {
            "Split": "Total",
            "N_estancias": n_total,
            "N_fallecidos": n_pos_total,
            "Prevalencia_pct": 100.0 * n_pos_total / max(n_total, 1),
        }
    )
    return pd.DataFrame(rows)


def make_preprocessors(feature_cols: list[str]) -> tuple[ColumnTransformer, ColumnTransformer]:
    numeric_transformer_linear = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    numeric_transformer_tree = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    preprocessor_linear = ColumnTransformer(
        transformers=[("num", numeric_transformer_linear, feature_cols)],
        remainder="drop",
    )

    preprocessor_tree = ColumnTransformer(
        transformers=[("num", numeric_transformer_tree, feature_cols)],
        remainder="drop",
    )

    return preprocessor_linear, preprocessor_tree


def build_model_spaces(random_state: int, n_pos: int, n_neg: int) -> dict[str, tuple[Pipeline, dict[str, Any]]]:
    pre_linear, pre_tree = make_preprocessors([])

    class_ratio = max(n_neg / max(n_pos, 1), 1.0)

    logistic_pipe = Pipeline(
        steps=[
            ("preprocess", pre_linear),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    random_state=random_state,
                ),
            ),
        ],
        memory=None,
    )
    logistic_space = {
        "model__penalty": ["l2", "l1"],
        "model__C": [0.01, 0.1, 1, 5, 10, 50],
        "model__solver": ["liblinear"],
        "model__class_weight": [None, "balanced"],
    }

    rf_pipe = Pipeline(
        steps=[
            ("preprocess", pre_tree),
            (
                "model",
                RandomForestClassifier(
                    random_state=random_state,
                    n_jobs=-1,
                    min_samples_leaf=1,
                    max_features="sqrt",
                ),
            ),
        ],
        memory=None,
    )
    rf_space = {
        "model__n_estimators": [200, 400, 800],
        "model__max_depth": [None, 6, 10, 16],
        "model__min_samples_split": [2, 10, 25],
        "model__min_samples_leaf": [1, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.5],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
    }

    model_spaces: dict[str, tuple[Pipeline, dict[str, Any]]] = {
        "logistic_regression": (logistic_pipe, logistic_space),
        "random_forest": (rf_pipe, rf_space),
    }

    xgb_classifier = resolve_xgb_classifier()
    if xgb_classifier is not None:
        xgb_pipe = Pipeline(
            steps=[
                ("preprocess", pre_tree),
                (
                    "model",
                    xgb_classifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ],
            memory=None,
        )
        xgb_space = {
            "model__n_estimators": [200, 400, 800],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [3, 4, 6, 8],
            "model__min_child_weight": [1, 3, 5],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_alpha": [0.0, 0.1, 1.0],
            "model__reg_lambda": [1.0, 5.0, 10.0],
            "model__scale_pos_weight": [1.0, round(class_ratio, 2), round(class_ratio * 1.5, 2)],
        }
        model_spaces["xgboost"] = (xgb_pipe, xgb_space)

    return model_spaces


def set_feature_columns(pipe: Pipeline, feature_cols: list[str]) -> Pipeline:
    preprocess = pipe.named_steps["preprocess"]
    preprocess.transformers = [("num", preprocess.transformers[0][1], feature_cols)]
    return pipe


def run_model_search(
    model_name: str,
    pipe: Pipeline,
    param_space: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int,
    cv: StratifiedKFold,
    random_state: int,
) -> RandomizedSearchCV:
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_space,
        n_iter=min(n_iter, max(1, int(np.prod([len(v) for v in param_space.values()]) if model_name == "logistic_regression" else n_iter))),
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
        verbose=0,
    )
    search.fit(x_train, y_train)
    return search


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    metrics = {
        "AUC_ROC": float(roc_auc_score(y_true, y_score)),
        "Brier_score": float(brier_score_loss(y_true, y_score)),
    }
    metrics.update(ranking_metrics(y_true, y_score))
    return metrics


def save_roc_figure(curves: list[dict[str, Any]], output_path: Path) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 6))
    for item in curves:
        ax.plot(item["fpr"], item["tpr"], label=f"{item['model']} (AUC={item['auc']:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Azar (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curvas ROC comparadas (Set A test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_calibration_figure(calibrations: list[dict[str, Any]], output_path: Path) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 6))
    for item in calibrations:
        ax.plot(item["pred"], item["obs"], marker="o", label=item["model"])

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Calibración perfecta")
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia observada")
    ax.set_title("Curvas de calibración (Set A test)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_score_distribution(y_true: np.ndarray, y_score: np.ndarray, model_name: str, output_path: Path) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_score[y_true == 0], bins=35, alpha=0.6, density=True, label="No fallece", color="#4c78a8")
    ax.hist(y_score[y_true == 1], bins=35, alpha=0.6, density=True, label="Fallece", color="#e45756")
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Densidad")
    ax.set_title(f"Distribución de scores - {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def format_result_tables(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prob_cols = ["Model", "AUC_ROC", "Brier_score"]
    prob_table = results_df[prob_cols].copy().sort_values("AUC_ROC", ascending=False)

    ranking_cols = ["Model", *RECALL_COLS]
    ranking_table = results_df[ranking_cols].copy().sort_values(RECALL_100, ascending=False)

    purity_cols = ["Model", *PURITY_COLS]
    purity_table = results_df[purity_cols].copy().sort_values(PURITY_100, ascending=False)

    baseline_row = pd.DataFrame([{"Model": "baseline_hito2", **BASELINE_HITO2}])

    prob_table = pd.concat([baseline_row[["Model", "AUC_ROC", "Brier_score"]], prob_table], ignore_index=True)
    ranking_table = pd.concat([baseline_row[["Model", *RECALL_COLS]], ranking_table], ignore_index=True)
    purity_table = pd.concat(
        [
            baseline_row[["Model", *PURITY_COLS]],
            purity_table,
        ],
        ignore_index=True,
    )

    return prob_table, ranking_table, purity_table


def run_experiments(project_root: Path, random_state: int, n_iter: int, force_rebuild: bool = False) -> dict[str, Any]:
    paths = resolve_paths(project_root)
    ensure_dirs(paths)

    data, data_source = load_or_build_dataset(paths, force_rebuild=force_rebuild)

    required = {"RecordID", "In-hospital_death"}
    if not required.issubset(set(data.columns)):
        raise ValueError("El dataset procesado no contiene RecordID e In-hospital_death.")

    work_df = data.copy()
    for col in work_df.columns:
        if col not in ["RecordID", "In-hospital_death"]:
            work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    train_df, valid_df, test_df = split_dataset(work_df, random_state=random_state)

    split_df = split_summary(train_df, valid_df, test_df)
    split_df.to_csv(paths["tables_dir"] / "table_6_1_split_distribution.csv", index=False)

    feature_reuse_df = build_feature_reuse_table(work_df)
    feature_reuse_df.to_csv(paths["tables_dir"] / "table_feature_reuse_hito2_vs_hito3.csv", index=False)

    train_valid_df = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)

    feature_cols = [c for c in train_valid_df.columns if c not in ["RecordID", "In-hospital_death"]]
    x_train_valid = train_valid_df[feature_cols]
    y_train_valid = train_valid_df["In-hospital_death"].astype(int)

    x_test = test_df[feature_cols]
    y_test = test_df["In-hospital_death"].astype(int).to_numpy()

    n_pos = int(y_train_valid.sum())
    n_neg = int(len(y_train_valid) - n_pos)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    model_spaces = build_model_spaces(random_state=random_state, n_pos=n_pos, n_neg=n_neg)

    all_results: list[dict[str, Any]] = []
    roc_items: list[dict[str, Any]] = []
    calibration_items: list[dict[str, Any]] = []
    best_params: dict[str, Any] = {}

    baseline_metrics, baseline_pred_df, baseline_used_features = evaluate_baseline_hito2_replicated(
        train_valid_df=train_valid_df,
        test_df=test_df,
    )
    baseline_row = {"Model": "baseline_hito2_replicated_60_20_20", **baseline_metrics}
    all_results.append(baseline_row)
    baseline_pred_df.to_csv(paths["predictions_dir"] / "predictions_baseline_hito2_replicated_60_20_20.csv", index=False)

    fpr_b, tpr_b, _ = roc_curve(y_test, baseline_pred_df["y_pred_proba"].to_numpy())
    roc_items.append({"model": "baseline_hito2_replicated_60_20_20", "fpr": fpr_b, "tpr": tpr_b, "auc": baseline_metrics["AUC_ROC"]})
    obs_b, pred_b = calibration_curve(y_test, baseline_pred_df["y_pred_proba"].to_numpy(), n_bins=10, strategy="quantile")
    calibration_items.append({"model": "baseline_hito2_replicated_60_20_20", "obs": obs_b, "pred": pred_b})
    save_score_distribution(
        y_true=y_test,
        y_score=baseline_pred_df["y_pred_proba"].to_numpy(),
        model_name="baseline_hito2_replicated_60_20_20",
        output_path=paths["figures_dir"] / "scores_distribution_baseline_hito2_replicated_60_20_20.png",
    )

    for model_name, (pipe, param_space) in model_spaces.items():
        pipe = set_feature_columns(pipe, feature_cols)

        search = run_model_search(
            model_name=model_name,
            pipe=pipe,
            param_space=param_space,
            x_train=x_train_valid,
            y_train=y_train_valid,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
        )

        best_model = search.best_estimator_
        y_score = best_model.predict_proba(x_test)[:, 1]

        metrics = evaluate_predictions(y_true=y_test, y_score=y_score)
        result_row = {"Model": model_name, **metrics}
        all_results.append(result_row)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_items.append({"model": model_name, "fpr": fpr, "tpr": tpr, "auc": metrics["AUC_ROC"]})

        obs, pred = calibration_curve(y_test, y_score, n_bins=10, strategy="quantile")
        calibration_items.append({"model": model_name, "obs": obs, "pred": pred})

        pred_df = pd.DataFrame(
            {
                "RecordID": test_df["RecordID"].to_numpy(),
                "y_true": y_test,
                "y_pred_proba": y_score,
            }
        ).sort_values("y_pred_proba", ascending=False)
        pred_df.to_csv(paths["predictions_dir"] / f"predictions_{model_name}.csv", index=False)

        save_score_distribution(
            y_true=y_test,
            y_score=y_score,
            model_name=model_name,
            output_path=paths["figures_dir"] / f"scores_distribution_{model_name}.png",
        )

        best_params[model_name] = {
            "best_cv_auc": float(search.best_score_),
            "best_params": search.best_params_,
        }

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(paths["tables_dir"] / "table_results_all_metrics.csv", index=False)

    prob_table, ranking_table, purity_table = format_result_tables(results_df)
    prob_table.to_csv(paths["tables_dir"] / "table_6_2_auc_brier.csv", index=False)
    ranking_table.to_csv(paths["tables_dir"] / "table_6_3_recall_at_k.csv", index=False)
    purity_table.to_csv(paths["tables_dir"] / "table_6_4_pct_top_k.csv", index=False)

    best_model_row = results_df.sort_values(["AUC_ROC", "Brier_score"], ascending=[False, True]).iloc[0]
    summary_table = pd.DataFrame(
        [
            {
                "Metrica": "AUC-ROC",
                "Baseline_Hito2": BASELINE_HITO2["AUC_ROC"],
                "Mejor_modelo_Hito3": float(best_model_row["AUC_ROC"]),
                "Ganancia": float(best_model_row["AUC_ROC"] - BASELINE_HITO2["AUC_ROC"]),
            },
            {
                "Metrica": "Brier",
                "Baseline_Hito2": BASELINE_HITO2["Brier_score"],
                "Mejor_modelo_Hito3": float(best_model_row["Brier_score"]),
                "Ganancia": float(BASELINE_HITO2["Brier_score"] - best_model_row["Brier_score"]),
            },
            {
                "Metrica": "Recall@50",
                "Baseline_Hito2": BASELINE_HITO2[RECALL_50],
                "Mejor_modelo_Hito3": float(best_model_row[RECALL_50]),
                "Ganancia": float(best_model_row[RECALL_50] - BASELINE_HITO2[RECALL_50]),
            },
            {
                "Metrica": "%fallecidos_top50",
                "Baseline_Hito2": BASELINE_HITO2[PURITY_50],
                "Mejor_modelo_Hito3": float(best_model_row[PURITY_50]),
                "Ganancia": float(best_model_row[PURITY_50] - BASELINE_HITO2[PURITY_50]),
            },
        ]
    )
    summary_table.to_csv(paths["tables_dir"] / "table_6_5_summary_gains.csv", index=False)

    save_roc_figure(roc_items, paths["figures_dir"] / "roc_comparison_models.png")
    save_calibration_figure(calibration_items, paths["figures_dir"] / "calibration_comparison_models.png")

    manifest = {
        "dataset_source": data_source,
        "dataset_rows": int(len(work_df)),
        "dataset_columns": int(work_df.shape[1]),
        "prevalence": float(work_df["In-hospital_death"].mean()),
        "baseline_hito2_features_expected": BASELINE_HITO2_FEATURES,
        "baseline_hito2_features_used": baseline_used_features,
        "random_state": random_state,
        "n_iter": n_iter,
        "models_ran": [row["Model"] for row in all_results],
        "best_params": best_params,
        "tables_dir": str(paths["tables_dir"]),
        "figures_dir": str(paths["figures_dir"]),
        "predictions_dir": str(paths["predictions_dir"]),
    }

    with open(paths["hito3_root"] / "artifacts" / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimentos Hito 3 - MIMIC-TRIAGE")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Ruta del proyecto (carpeta que contiene Hito 2 y Hito 3)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--n-iter", type=int, default=30, help="Iteraciones de RandomizedSearchCV por modelo")
    parser.add_argument("--force-rebuild", action="store_true", help="Fuerza reconstrucción del CSV desde RAW")

    args = parser.parse_args()

    manifest = run_experiments(
        project_root=Path(args.project_root),
        random_state=args.seed,
        n_iter=args.n_iter,
        force_rebuild=args.force_rebuild,
    )

    print("Experimentos completados.")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
