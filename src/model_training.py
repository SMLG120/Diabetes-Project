"""
=============================================================================
SAMUEL — MODEL ENGINEER: model_training.py
=============================================================================
Responsibilities:
  - Baseline  : Logistic Regression
  - Intermediate: Random Forest
  - Advanced  : LightGBM, XGBoost, CatBoost
  - Stratified K-Fold cross-validation (k=5)
  - Class-imbalance handling (class_weight / scale_pos_weight)
  - Hyperparameter tuning with Optuna
  - Save best model + all OOF predictions
=============================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics         import roc_auc_score
import lightgbm as lgb
import xgboost  as xgb
import catboost as cb

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_SPLITS     = 5
np.random.seed(RANDOM_STATE)


# ── 1. STRATIFIED K-FOLD CV EVALUATOR ────────────────────────────────────────

def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_SPLITS,
    model_name: str = "Model"
) -> tuple[float, float, np.ndarray]:
    """
    Run Stratified K-Fold cross-validation and collect OOF predictions.

    Args:
        model:      sklearn-compatible estimator
        X:          feature matrix
        y:          target vector
        n_splits:   number of CV folds
        model_name: display name

    Returns:
        (mean_auc, std_auc, oof_predictions)
    """
    skf       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y))
    fold_aucs = []

    print(f"\n[CV] {model_name} — {n_splits}-Fold Stratified CV")
    print("-" * 50)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        preds               = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx]  = preds
        auc                 = roc_auc_score(y_val, preds)
        fold_aucs.append(auc)
        print(f"  Fold {fold}: AUC = {auc:.5f}")

    mean_auc = np.mean(fold_aucs)
    std_auc  = np.std(fold_aucs)
    print(f"  ► Mean AUC: {mean_auc:.5f} ± {std_auc:.5f}")
    return mean_auc, std_auc, oof_preds


# ── 2. BASELINE: LOGISTIC REGRESSION ─────────────────────────────────────────

def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray
) -> tuple[LogisticRegression, float, np.ndarray]:
    """
    Logistic Regression baseline.
    Uses class_weight='balanced' to handle imbalance.
    Expects SCALED features.
    """
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE
    )
    mean_auc, std_auc, oof = cross_validate_model(model, X, y, model_name="LogisticRegression")

    # Refit on full train set
    model.fit(X, y)
    return model, mean_auc, oof


# ── 3. INTERMEDIATE: RANDOM FOREST ───────────────────────────────────────────

def train_random_forest(
    X: np.ndarray,
    y: np.ndarray
) -> tuple[RandomForestClassifier, float, np.ndarray]:
    """
    Random Forest with balanced class weights.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    mean_auc, std_auc, oof = cross_validate_model(model, X, y, model_name="RandomForest")
    model.fit(X, y)
    return model, mean_auc, oof


# ── 4. ADVANCED: LIGHTGBM ────────────────────────────────────────────────────

def train_lightgbm(
    X: np.ndarray,
    y: np.ndarray
) -> tuple[lgb.LGBMClassifier, float, np.ndarray]:
    """
    LightGBM — handles imbalance via is_unbalance=True.
    """
    # Compute scale ratio for imbalance
    neg, pos    = np.bincount(y)
    scale_ratio = neg / pos

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        min_child_samples=20,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=scale_ratio,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    mean_auc, std_auc, oof = cross_validate_model(model, X, y, model_name="LightGBM")
    model.fit(X, y)
    return model, mean_auc, oof


# ── 5. ADVANCED: XGBOOST ─────────────────────────────────────────────────────

def train_xgboost(
    X: np.ndarray,
    y: np.ndarray
) -> tuple[xgb.XGBClassifier, float, np.ndarray]:
    """
    XGBoost — scale_pos_weight handles imbalance.
    """
    neg, pos    = np.bincount(y)
    scale_ratio = neg / pos

    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        colsample_bytree=0.8,
        subsample=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_ratio,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
    mean_auc, std_auc, oof = cross_validate_model(model, X, y, model_name="XGBoost")
    model.fit(X, y)
    return model, mean_auc, oof


# ── 6. ADVANCED: CATBOOST ────────────────────────────────────────────────────

def train_catboost(
    X: np.ndarray,
    y: np.ndarray
) -> tuple[cb.CatBoostClassifier, float, np.ndarray]:
    """
    CatBoost — auto handles imbalance via auto_class_weights.
    """
    model = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        auto_class_weights="Balanced",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=0
    )
    mean_auc, std_auc, oof = cross_validate_model(model, X, y, model_name="CatBoost")
    model.fit(X, y)
    return model, mean_auc, oof


# ── 7. OPTUNA HYPERPARAMETER TUNING (LightGBM) ───────────────────────────────

def tune_lightgbm_optuna(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50
) -> lgb.LGBMClassifier:
    """
    Tune LightGBM hyperparameters using Optuna.
    Optimizes ROC-AUC via 5-fold CV.

    Args:
        X:        feature matrix
        y:        target
        n_trials: number of Optuna trials

    Returns:
        Best-fitted LGBMClassifier
    """
    neg, pos    = np.bincount(y)
    scale_ratio = neg / pos

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight":  scale_ratio,
            "random_state":      RANDOM_STATE,
            "n_jobs":            -1,
            "verbose":           -1,
        }
        model = lgb.LGBMClassifier(**params)
        skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        aucs  = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
        return aucs.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    print(f"\n[Optuna] Tuning LightGBM — {n_trials} trials …")
    start = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - start

    best_params = study.best_params
    best_auc    = study.best_value
    print(f"[Optuna] Best AUC : {best_auc:.5f}  ({elapsed:.1f}s)")
    print(f"[Optuna] Best params: {best_params}")

    # Refit best model on full data
    best_model = lgb.LGBMClassifier(
        **best_params,
        scale_pos_weight=scale_ratio,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    best_model.fit(X, y)
    return best_model


# ── 8. SAVE / LOAD MODELS ─────────────────────────────────────────────────────

def save_model(model, name: str, directory: str = "models/") -> str:
    """Persist a fitted model with joblib."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"[Save]  Model saved → {path}")
    return path


def load_model(path: str):
    """Load a persisted model."""
    model = joblib.load(path)
    print(f"[Load]  Model loaded ← {path}")
    return model


# ── 9. MASTER TRAINING PIPELINE ──────────────────────────────────────────────

def run_training_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_train_scaled: np.ndarray,
    tune: bool = False,
    n_trials: int = 50
) -> dict:
    """
    Train all models, collect CV scores and OOF predictions.

    Args:
        X_train:        raw (unscaled) features — for tree models
        y_train:        target labels
        X_train_scaled: scaled features — for Logistic Regression
        tune:           if True, run Optuna tuning on LightGBM
        n_trials:       Optuna trials count

    Returns:
        dict with keys: models, cv_scores, oof_predictions
    """
    results = {"models": {}, "cv_scores": {}, "oof_predictions": {}}

    # ── Logistic Regression (scaled features) ─────────────────────────────
    lr, lr_auc, lr_oof = train_logistic_regression(X_train_scaled, y_train)
    results["models"]["logistic_regression"]        = lr
    results["cv_scores"]["logistic_regression"]     = lr_auc
    results["oof_predictions"]["logistic_regression"] = lr_oof
    save_model(lr, "logistic_regression")

    # ── Random Forest (raw features) ──────────────────────────────────────
    rf, rf_auc, rf_oof = train_random_forest(X_train, y_train)
    results["models"]["random_forest"]        = rf
    results["cv_scores"]["random_forest"]     = rf_auc
    results["oof_predictions"]["random_forest"] = rf_oof
    save_model(rf, "random_forest")

    # ── LightGBM ──────────────────────────────────────────────────────────
    if tune:
        lgbm_model = tune_lightgbm_optuna(X_train, y_train, n_trials=n_trials)
        lgbm_auc, _, lgbm_oof = cross_validate_model(
            lgbm_model, X_train, y_train, model_name="LightGBM (tuned)")
    else:
        lgbm_model, lgbm_auc, lgbm_oof = train_lightgbm(X_train, y_train)
    results["models"]["lightgbm"]        = lgbm_model
    results["cv_scores"]["lightgbm"]     = lgbm_auc
    results["oof_predictions"]["lightgbm"] = lgbm_oof
    save_model(lgbm_model, "lightgbm")

    # ── XGBoost ───────────────────────────────────────────────────────────
    xgb_model, xgb_auc, xgb_oof = train_xgboost(X_train, y_train)
    results["models"]["xgboost"]        = xgb_model
    results["cv_scores"]["xgboost"]     = xgb_auc
    results["oof_predictions"]["xgboost"] = xgb_oof
    save_model(xgb_model, "xgboost")

    # ── CatBoost ──────────────────────────────────────────────────────────
    cat_model, cat_auc, cat_oof = train_catboost(X_train, y_train)
    results["models"]["catboost"]        = cat_model
    results["cv_scores"]["catboost"]     = cat_auc
    results["oof_predictions"]["catboost"] = cat_oof
    save_model(cat_model, "catboost")

    # ── Leaderboard ───────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("  MODEL COMPARISON (Cross-Validation ROC-AUC)")
    print("="*50)
    scores = pd.Series(results["cv_scores"]).sort_values(ascending=False)
    for name, score in scores.items():
        tag = "  ◄ BEST" if name == scores.index[0] else ""
        print(f"  {name:<25}  {score:.5f}{tag}")

    best_name = scores.index[0]
    print(f"\n[Best Model] → {best_name}")
    results["best_model_name"] = best_name

    return results


# ── EXAMPLE USAGE ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # This block runs when script is executed directly.
    # Replace with actual data from preprocessing pipeline.
    from preprocessing import run_preprocessing_pipeline

    data = run_preprocessing_pipeline()
    results = run_training_pipeline(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_train_scaled=data["X_train_scaled"],
        tune=False   # Set True for Optuna tuning (slower)
    )
    print("\n[model_training.py] Done. Best:", results["best_model_name"])
