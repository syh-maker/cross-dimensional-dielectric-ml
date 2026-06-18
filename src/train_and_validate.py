import argparse
import os
import re
import sys
import warnings
from typing import Dict, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from ml_types import DropCollinearFeatures, PreprocessBundle

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_MODEL_DIR = "models"
DEFAULT_TRAIN_X = "data/processed/feats_x.csv"
DEFAULT_TRAIN_Y = "data/processed/labels_y.csv"
DEFAULT_VAL_X = "data/processed/external_validation/validation_features_X.csv"
DEFAULT_VAL_Y = "data/processed/external_validation/validation_labels_y.csv"

TARGETS = ["e_electronic", "e_ionic", "e_total"]

OUTER_N_SPLITS = 5
INNER_N_SPLITS = 5
OUTER_RANDOM_STATE = 42
INNER_RANDOM_STATE = 123
TEST_SIZE = 0.10
TEST_RANDOM_STATE = 42

ENSEMBLE_N_SEEDS = 10
ENSEMBLE_SEED_BASE = 42
ENSEMBLE_SEED_STEP = 17

SHAP_RANK_TOPN = 100
INNER_SHAP_AVG_RUNS = 20
FINAL_SHAP_AVG_RUNS = 30
OPTUNA_TRIALS = 120

TARGET_CONFIGS = {
    "e_electronic": {"name": "log", "trans": "log", "weight": "mild_boost", "rfe_start": SHAP_RANK_TOPN},
    "e_ionic": {"name": "log", "trans": "log", "weight": "high_boost", "rfe_start": SHAP_RANK_TOPN},
    "e_total": {"name": "log", "trans": "log", "weight": "mild_boost", "rfe_start": SHAP_RANK_TOPN},
}

SHAP_BASE_PARAMS = {
    "n_estimators": 160,
    "max_depth": 3,
    "learning_rate": 0.04,
    "subsample": 0.72,
    "colsample_bytree": 0.72,
    "min_child_weight": 6,
    "reg_alpha": 0.2,
    "reg_lambda": 6.0,
    "gamma": 0.0,
    "tree_method": "hist",
    "n_jobs": -1,
}

FS_EVAL_PARAMS = {
    "n_estimators": 180,
    "max_depth": 3,
    "learning_rate": 0.04,
    "subsample": 0.72,
    "colsample_bytree": 0.72,
    "min_child_weight": 6,
    "reg_alpha": 0.2,
    "reg_lambda": 8.0,
    "gamma": 0.0,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
}


def normalize_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = [re.sub(r"[^a-zA-Z0-9]", "_", str(c)).strip("_") for c in out.columns]
    cols = ["f_" + c if c and c.isdigit() else c for c in cols]
    out.columns = cols
    return out


def inject_quantum_polarizability(df_raw: pd.DataFrame) -> List[float]:
    proxy_col = None
    for candidate in ["MagpieData_mean_Number", "MagpieData_mean_AtomicWeight", "MagpieData_mean_CovalentRadius"]:
        if candidate in df_raw.columns:
            proxy_col = candidate
            break
    if proxy_col is None:
        return [0.0] * len(df_raw)
    z_mean = pd.to_numeric(df_raw[proxy_col], errors="coerce").fillna(10.0)
    return ((z_mean / 10.0) ** 1.3).tolist()


def transform_label(values, config: Dict) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if config["trans"] == "log":
        return np.log1p(np.maximum(values, 0.0))
    return values


def inverse_transform(values, config: Dict) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if config["trans"] == "log":
        return np.expm1(np.clip(values, -10.0, 20.0))
    return values


def get_weight_vector(y: pd.Series, config_type: str) -> pd.Series:
    y_values = pd.Series(np.asarray(y, dtype=float), index=y.index)
    weights = np.ones(len(y_values), dtype=float)
    if config_type == "mild_boost":
        weights = np.where(y_values.values > 2.5, weights * 1.5, weights)
        weights = np.clip(weights, 1.0, 5.0)
    if config_type == "high_boost":
        p75 = np.percentile(y_values.values, 75)
        p90 = np.percentile(y_values.values, 90)
        weights = np.where(y_values.values > p90, weights * 4.0, np.where(y_values.values > p75, weights * 2.0, weights))
        weights = np.clip(weights, 1.0, 20.0)
    return pd.Series(weights, index=y.index)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) == 0:
        return {"R2": np.nan, "RMSE": np.nan, "N": 0}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan
    return {"R2": r2, "RMSE": rmse, "N": int(len(y_true))}


def build_feature_counts(start_n: int) -> List[int]:
    start_n = min(start_n, SHAP_RANK_TOPN)
    counts = []
    current = start_n
    while current > 40:
        counts.append(current)
        current -= 5
    while current >= 10:
        counts.append(current)
        current -= 1
    counts = sorted(set(c for c in counts if 10 <= c <= start_n), reverse=True)
    if start_n >= 10 and 10 not in counts:
        counts.append(10)
    return counts


def build_ensemble_seeds(n_models: int = ENSEMBLE_N_SEEDS) -> List[int]:
    return [ENSEMBLE_SEED_BASE + i * ENSEMBLE_SEED_STEP for i in range(n_models)]


def safe_align(df_source: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    out = df_source.copy()
    for col in target_cols:
        if col not in out.columns:
            out[col] = 0.0
    return out[target_cols]


def fit_preprocessor(X_raw: pd.DataFrame) -> Tuple[PreprocessBundle, pd.DataFrame]:
    X = X_raw.copy().replace([np.inf, -np.inf], np.nan)
    input_cols = list(X.columns)
    non_empty_cols = [col for col in X.columns if not X[col].isna().all()]
    X = X[non_empty_cols]
    imputer = SimpleImputer(strategy="median")
    Xi = pd.DataFrame(imputer.fit_transform(X), columns=non_empty_cols, index=X.index)
    variance = VarianceThreshold(0.0)
    Xv_array = variance.fit_transform(Xi)
    variance_cols = list(Xi.columns[variance.get_support()])
    Xv = pd.DataFrame(Xv_array, columns=variance_cols, index=X.index)
    scaler = RobustScaler()
    Xs = pd.DataFrame(scaler.fit_transform(Xv), columns=variance_cols, index=X.index)
    dropper = DropCollinearFeatures(0.99)
    dropper.fit(Xs)
    Xf = dropper.transform(Xs)
    bundle = PreprocessBundle(input_cols, non_empty_cols, imputer, variance, variance_cols, scaler, variance_cols, dropper, list(Xf.columns))
    return bundle, Xf


def apply_preprocessor(bundle: PreprocessBundle, X_raw: pd.DataFrame) -> pd.DataFrame:
    X_aligned = safe_align(X_raw, bundle.input_cols).replace([np.inf, -np.inf], np.nan)
    X = X_aligned[bundle.non_empty_cols]
    Xi = pd.DataFrame(bundle.imputer.transform(X), columns=bundle.non_empty_cols, index=X_raw.index)
    Xv = pd.DataFrame(bundle.variance.transform(Xi), columns=bundle.variance_cols, index=X_raw.index)
    Xs = pd.DataFrame(bundle.scaler.transform(Xv), columns=bundle.final_cols_before_drop, index=X_raw.index)
    Xf = bundle.dropper.transform(Xs)
    return Xf.reindex(columns=bundle.output_cols, fill_value=0.0)


def shap_rank_features(X_train: pd.DataFrame, y_train: pd.Series, w_train: pd.Series, n_runs: int) -> Tuple[List[str], pd.DataFrame]:
    cols = X_train.columns.tolist()
    importances = []
    for seed in range(n_runs):
        params = SHAP_BASE_PARAMS.copy()
        params["random_state"] = 1000 + seed
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, sample_weight=w_train)
        shap_values = np.asarray(shap.TreeExplainer(model).shap_values(X_train), dtype=float)
        importances.append(pd.Series(np.abs(shap_values).mean(axis=0), index=cols))
    importance_wide = pd.concat(importances, axis=1)
    importance_df = pd.DataFrame({"Feature": cols, "MeanAbsSHAP": importance_wide.mean(axis=1).values, "StdAbsSHAP": importance_wide.std(axis=1).values})
    importance_df = importance_df.sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)
    importance_df["Rank"] = importance_df.index + 1
    return importance_df["Feature"].tolist(), importance_df


def evaluate_prefix_feature_counts(X_train: pd.DataFrame, y_train: pd.Series, w_train: pd.Series, ranked_feats: List[str], n_splits: int, random_state: int) -> Tuple[List[str], float, pd.DataFrame]:
    ranked_feats = ranked_feats[:SHAP_RANK_TOPN]
    feature_counts = build_feature_counts(len(ranked_feats))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_score = -np.inf
    best_feats = ranked_feats.copy()
    records = []
    for n_feats in feature_counts:
        current_feats = ranked_feats[:n_feats]
        scores = []
        for train_idx, valid_idx in cv.split(X_train):
            X_tr = X_train.iloc[train_idx][current_feats]
            X_va = X_train.iloc[valid_idx][current_feats]
            y_tr = y_train.iloc[train_idx]
            y_va = y_train.iloc[valid_idx]
            w_tr = w_train.iloc[train_idx]
            model = XGBRegressor(**FS_EVAL_PARAMS)
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            pred = model.predict(X_va)
            scores.append(r2_score(y_va, pred))
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        records.append({"n_features": n_feats, "cv_mean_r2": mean_score, "cv_std_r2": std_score})
        if mean_score > best_score:
            best_score = mean_score
            best_feats = current_feats.copy()
    return best_feats, best_score, pd.DataFrame(records)


def tune_xgb_with_optuna(X_train: pd.DataFrame, y_train: pd.Series, w_train: pd.Series, n_splits: int, random_state: int, n_trials: int) -> Tuple[Dict, float]:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(cv.split(X_train))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 120, 450),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 0.85),
            "subsample": trial.suggest_float("subsample", 0.55, 0.85),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.05, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 40.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 4, 16),
            "gamma": trial.suggest_float("gamma", 0.0, 1.5),
            "random_state": 42,
            "tree_method": "hist",
            "n_jobs": -1,
        }
        scores = []
        for train_idx, valid_idx in splits:
            X_tr = X_train.iloc[train_idx]
            X_va = X_train.iloc[valid_idx]
            y_tr = y_train.iloc[train_idx]
            y_va = y_train.iloc[valid_idx]
            w_tr = w_train.iloc[train_idx]
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            pred = model.predict(X_va)
            scores.append(r2_score(y_va, pred))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    best_params = study.best_params.copy()
    best_params.update({"random_state": 42, "tree_method": "hist", "n_jobs": -1})
    return best_params, float(study.best_value)


def fit_seed_ensemble(X_train: pd.DataFrame, y_train: pd.Series, sample_weight: pd.Series, base_params: Dict) -> List[XGBRegressor]:
    models = []
    for seed in build_ensemble_seeds():
        params = base_params.copy()
        params.update({"random_state": int(seed), "tree_method": "hist", "n_jobs": -1})
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        models.append(model)
    return models


def predict_seed_ensemble(models: List[XGBRegressor], X_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    pred_matrix = np.column_stack([model.predict(X_data) for model in models])
    return pred_matrix.mean(axis=1), pred_matrix.std(axis=1)


def run_outer_cv(X_curr: pd.DataFrame, y_phys: pd.Series, config: Dict, target: str, n_trials: int) -> Dict:
    y_trans_all = pd.Series(transform_label(y_phys.values, config), index=y_phys.index)
    outer_cv = KFold(n_splits=OUTER_N_SPLITS, shuffle=True, random_state=OUTER_RANDOM_STATE)
    oof_pred_trans = pd.Series(index=y_phys.index, dtype=float)
    oof_pred_phys = pd.Series(index=y_phys.index, dtype=float)
    fold_records = []
    feature_counter = {}
    for fold_id, (train_pos, valid_pos) in enumerate(outer_cv.split(X_curr), start=1):
        train_idx = y_phys.index[train_pos]
        valid_idx = y_phys.index[valid_pos]
        X_tr_raw = X_curr.loc[train_idx].copy()
        X_va_raw = X_curr.loc[valid_idx].copy()
        y_tr_phys = y_phys.loc[train_idx]
        y_va_phys = y_phys.loc[valid_idx]
        y_tr_trans = pd.Series(transform_label(y_tr_phys.values, config), index=train_idx)
        y_va_trans = pd.Series(transform_label(y_va_phys.values, config), index=valid_idx)
        w_tr = get_weight_vector(y_tr_phys, config["weight"])
        bundle, X_tr = fit_preprocessor(X_tr_raw)
        X_va = apply_preprocessor(bundle, X_va_raw)
        ranked_feats, _ = shap_rank_features(X_tr, y_tr_trans, w_tr, INNER_SHAP_AVG_RUNS)
        ranked_feats = ranked_feats[:config["rfe_start"]]
        best_feats, best_fs_score, _ = evaluate_prefix_feature_counts(X_tr, y_tr_trans, w_tr, ranked_feats, INNER_N_SPLITS, INNER_RANDOM_STATE + fold_id)
        best_params, best_cv_r2 = tune_xgb_with_optuna(X_tr[best_feats], y_tr_trans, w_tr, INNER_N_SPLITS, 200 + fold_id, n_trials)
        model = XGBRegressor(**best_params)
        model.fit(X_tr[best_feats], y_tr_trans, sample_weight=w_tr)
        pred_trans = model.predict(X_va[best_feats])
        pred_phys = inverse_transform(pred_trans, config)
        oof_pred_trans.loc[valid_idx] = pred_trans
        oof_pred_phys.loc[valid_idx] = pred_phys
        fold_trans_metrics = compute_metrics(y_va_trans.values, pred_trans)
        fold_phys_metrics = compute_metrics(y_va_phys.values, pred_phys)
        for feat in best_feats:
            feature_counter[feat] = feature_counter.get(feat, 0) + 1
        fold_records.append({
            "target": target,
            "fold": fold_id,
            "n_features": len(best_feats),
            "feature_count_cv_r2": best_fs_score,
            "optuna_cv_r2": best_cv_r2,
            "outer_r2_transformed": fold_trans_metrics["R2"],
            "outer_rmse_transformed": fold_trans_metrics["RMSE"],
            "outer_r2_physical": fold_phys_metrics["R2"],
            "outer_rmse_physical": fold_phys_metrics["RMSE"],
        })
        print(f"{target} fold {fold_id:02d}: n_features={len(best_feats)}, outer_log_r2={fold_trans_metrics['R2']:.4f}, outer_physical_r2={fold_phys_metrics['R2']:.4f}")
    feature_stability = pd.DataFrame(sorted(feature_counter.items(), key=lambda item: (-item[1], item[0])), columns=["Feature", "Selected_in_Folds"])
    return {
        "oof_pred_trans": oof_pred_trans,
        "oof_pred_phys": oof_pred_phys,
        "outer_trans_metrics": compute_metrics(y_trans_all.values, oof_pred_trans.values),
        "outer_phys_metrics": compute_metrics(y_phys.values, oof_pred_phys.values),
        "fold_records": pd.DataFrame(fold_records),
        "feature_stability": feature_stability,
    }


def train_final_model(X_dev_raw, y_dev_phys, X_test_raw, y_test_phys, X_ext_raw, y_ext_phys, config, target, n_trials):
    y_dev_trans = pd.Series(transform_label(y_dev_phys.values, config), index=y_dev_phys.index)
    w_dev = get_weight_vector(y_dev_phys, config["weight"])
    bundle, X_dev = fit_preprocessor(X_dev_raw)
    X_test = apply_preprocessor(bundle, X_test_raw)
    X_ext = apply_preprocessor(bundle, X_ext_raw)
    ranked_feats, shap_importance = shap_rank_features(X_dev, y_dev_trans, w_dev, FINAL_SHAP_AVG_RUNS)
    ranked_feats = ranked_feats[:config["rfe_start"]]
    best_feats, feature_count_cv_r2, feature_count_history = evaluate_prefix_feature_counts(X_dev, y_dev_trans, w_dev, ranked_feats, INNER_N_SPLITS, INNER_RANDOM_STATE)
    best_params, optuna_cv_r2 = tune_xgb_with_optuna(X_dev[best_feats], y_dev_trans, w_dev, INNER_N_SPLITS, 333, n_trials)
    models = fit_seed_ensemble(X_dev[best_feats], y_dev_trans, w_dev, best_params)
    train_pred_trans, train_pred_std = predict_seed_ensemble(models, X_dev[best_feats])
    test_pred_trans, test_pred_std = predict_seed_ensemble(models, X_test[best_feats])
    ext_pred_trans, ext_pred_std = predict_seed_ensemble(models, X_ext[best_feats])
    train_pred_phys = inverse_transform(train_pred_trans, config)
    test_pred_phys = inverse_transform(test_pred_trans, config)
    ext_pred_phys = inverse_transform(ext_pred_trans, config)
    train_trans_metrics = compute_metrics(y_dev_trans.values, train_pred_trans)
    train_phys_metrics = compute_metrics(y_dev_phys.values, train_pred_phys)
    y_test_trans = transform_label(y_test_phys.values, config)
    test_trans_metrics = compute_metrics(y_test_trans, test_pred_trans)
    test_phys_metrics = compute_metrics(y_test_phys.values, test_pred_phys)
    y_ext_trans = transform_label(y_ext_phys.values, config)
    ext_trans_metrics = compute_metrics(y_ext_trans, ext_pred_trans)
    ext_phys_metrics = compute_metrics(y_ext_phys.values, ext_pred_phys)
    artifact = {
        "preprocess_bundle": bundle,
        "features": best_feats,
        "models": models,
        "config": config,
        "target": target,
        "best_params": best_params,
        "feature_count_cv_r2": feature_count_cv_r2,
        "optuna_cv_r2": optuna_cv_r2,
        "ensemble_seeds": build_ensemble_seeds(),
    }
    return {
        "artifact": artifact,
        "best_features": best_feats,
        "best_params": best_params,
        "feature_count_history": feature_count_history,
        "shap_importance": shap_importance,
        "train_pred_trans": train_pred_trans,
        "train_pred_phys": train_pred_phys,
        "train_pred_std": train_pred_std,
        "test_pred_trans": test_pred_trans,
        "test_pred_phys": test_pred_phys,
        "test_pred_std": test_pred_std,
        "ext_pred_trans": ext_pred_trans,
        "ext_pred_phys": ext_pred_phys,
        "ext_pred_std": ext_pred_std,
        "train_trans_metrics": train_trans_metrics,
        "train_phys_metrics": train_phys_metrics,
        "test_trans_metrics": test_trans_metrics,
        "test_phys_metrics": test_phys_metrics,
        "ext_trans_metrics": ext_trans_metrics,
        "ext_phys_metrics": ext_phys_metrics,
        "feature_count_cv_r2": feature_count_cv_r2,
        "optuna_cv_r2": optuna_cv_r2,
    }


def prediction_frame(material_ids, target, predictions) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "material_id": np.asarray(material_ids, dtype=object),
            f"pred_{target}": np.asarray(predictions, dtype=float),
        }
    )


def finalize_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    direct_columns = [f"pred_{target}" for target in TARGETS]
    missing = [column for column in direct_columns if column not in df.columns]
    if missing:
        raise RuntimeError(f"Missing prediction columns: {', '.join(missing)}")

    result = df[["material_id", *direct_columns]].copy()
    result["pred_e_total_sum"] = (
        result["pred_e_electronic"] + result["pred_e_ionic"]
    )
    return result


def apply_external_physical_constraints(
    target: str, features: pd.DataFrame, predictions
) -> np.ndarray:
    constrained = np.asarray(predictions, dtype=float).copy()
    elemental_range_column = "MagpieData_range_Number"
    if target == "e_ionic" and elemental_range_column in features.columns:
        elemental_mask = np.isclose(
            pd.to_numeric(features[elemental_range_column], errors="coerce"), 0.0
        )
        constrained[elemental_mask] = 0.0
    return constrained


def load_data(train_x_path, train_y_path, val_x_path, val_y_path):
    X_raw = normalize_names(pd.read_csv(train_x_path)).reset_index(drop=True)
    y_raw = pd.read_csv(train_y_path).reset_index(drop=True)
    X_raw["phys_Quantum_EAP"] = inject_quantum_polarizability(X_raw)
    min_len = min(len(X_raw), len(y_raw))
    X_raw = X_raw.iloc[:min_len].copy()
    y_raw = y_raw.iloc[:min_len].copy()
    X = X_raw.select_dtypes(include=[np.number]).copy()
    material_ids = y_raw.iloc[:, 0].astype(str).values
    X_val_raw = normalize_names(pd.read_csv(val_x_path))
    X_val_raw["phys_Quantum_EAP"] = inject_quantum_polarizability(X_val_raw)
    y_val_raw = pd.read_csv(val_y_path)
    x_id_col = X_val_raw.columns[0]
    y_id_col = y_val_raw.columns[0]
    val_merged = X_val_raw.merge(y_val_raw, left_on=x_id_col, right_on=y_id_col, how="inner")
    X_val_num = val_merged[X_val_raw.columns].select_dtypes(include=[np.number]).copy()
    y_val_df = val_merged[y_val_raw.columns].copy()
    val_material_ids = y_val_df.iloc[:, 0].astype(str).values
    return X, y_raw, material_ids, X_val_num, y_val_df, val_material_ids


def run_pipeline(args):
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.model_output, exist_ok=True)

    X, y_raw, material_ids, X_ext, y_ext_df, ext_material_ids = load_data(args.train_x, args.train_y, args.val_x, args.val_y)
    missing_targets = [
        target
        for target in TARGETS
        if target not in y_raw.columns or target not in y_ext_df.columns
    ]
    if missing_targets:
        raise ValueError(f"Missing target columns: {', '.join(missing_targets)}")

    n_samples = len(y_raw)
    all_indices = np.arange(n_samples)
    dev_indices, test_indices = train_test_split(all_indices, test_size=args.test_size, random_state=args.test_random_state, shuffle=True)
    dev_indices = np.array(sorted(dev_indices))
    test_indices = np.array(sorted(test_indices))
    dev_oof_df = pd.DataFrame({"material_id": material_ids[dev_indices]})
    test_df = pd.DataFrame({"material_id": material_ids[test_indices]})
    ext_df = pd.DataFrame({"material_id": ext_material_ids})

    for target in TARGETS:
        y_dev = y_raw.loc[dev_indices, target].dropna()
        if len(y_dev) < 30:
            raise ValueError(f"Not enough training rows for {target}")
        test_target_index = y_raw.loc[test_indices, target].dropna().index
        ext_target_index = y_ext_df[target].dropna().index
        config = TARGET_CONFIGS[target]
        X_dev = X.loc[y_dev.index].copy()
        X_test = X.loc[test_target_index].copy()
        y_test = y_raw.loc[test_target_index, target].copy()
        X_ext_target = X_ext.loc[ext_target_index].copy()
        y_ext = y_ext_df.loc[ext_target_index, target].copy()
        print(f"Training target: {target}")
        outer_result = run_outer_cv(X_dev, y_dev, config, target, args.trials)
        final_result = train_final_model(X_dev, y_dev, X_test, y_test, X_ext_target, y_ext, config, target, args.trials)
        joblib.dump(
            final_result["artifact"],
            os.path.join(args.model_output, f"{target}.joblib"),
        )

        dev_target_df = prediction_frame(
            material_ids[y_dev.index],
            target,
            outer_result["oof_pred_phys"].reindex(y_dev.index).values,
        )
        test_target_df = prediction_frame(
            material_ids[test_target_index],
            target,
            final_result["test_pred_phys"],
        )
        ext_predictions = apply_external_physical_constraints(
            target, X_ext_target, final_result["ext_pred_phys"]
        )
        ext_target_df = prediction_frame(
            ext_material_ids[ext_target_index],
            target,
            ext_predictions,
        )
        dev_oof_df = dev_oof_df.merge(dev_target_df, on="material_id", how="left")
        test_df = test_df.merge(test_target_df, on="material_id", how="left")
        ext_df = ext_df.merge(ext_target_df, on="material_id", how="left")

    finalize_prediction_table(dev_oof_df).to_csv(
        os.path.join(args.output, "development_oof_predictions.csv"), index=False
    )
    finalize_prediction_table(test_df).to_csv(
        os.path.join(args.output, "holdout_test_predictions.csv"), index=False
    )
    finalize_prediction_table(ext_df).to_csv(
        os.path.join(args.output, "external_validation_predictions.csv"), index=False
    )
    print(f"Saved results to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-x", default=DEFAULT_TRAIN_X)
    parser.add_argument("--train-y", default=DEFAULT_TRAIN_Y)
    parser.add_argument("--val-x", default=DEFAULT_VAL_X)
    parser.add_argument("--val-y", default=DEFAULT_VAL_Y)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-output", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--trials", type=int, default=OPTUNA_TRIALS)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--test-random-state", type=int, default=TEST_RANDOM_STATE)
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
