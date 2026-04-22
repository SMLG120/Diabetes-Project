"""
=============================================================================
TANEL — MLOPS / EVALUATION ENGINEER: evaluation.py
=============================================================================
Responsibilities:
  - Full pipeline integration
  - Evaluation metrics: ROC-AUC, Confusion Matrix, Precision/Recall
  - Ensemble methods: Averaging + Weighted Blending
  - CV vs Leaderboard gap analysis
  - Generate submission.csv for Kaggle
  - Final visualizations & report data
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
    classification_report
)

# ── Output directories ────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)


# ── 1. ROC-AUC EVALUATION ─────────────────────────────────────────────────────

def evaluate_roc_auc(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Model"
) -> float:
    """
    Compute and print ROC-AUC for a single model's OOF predictions.

    Args:
        y_true:       ground truth labels
        y_pred_proba: predicted probabilities (positive class)
        model_name:   display label

    Returns:
        roc_auc score
    """
    score = roc_auc_score(y_true, y_pred_proba)
    print(f"[AUC] {model_name:<25} → {score:.5f}")
    return score


# ── 2. CONFUSION MATRIX ───────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Model",
    threshold: float = 0.5,
    save: bool = True
) -> None:
    """
    Plot and (optionally) save the confusion matrix.

    Args:
        y_true:       ground truth labels
        y_pred_proba: predicted probabilities
        model_name:   title label
        threshold:    classification threshold
        save:         whether to write figure to disk
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp    = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Diabetes", "Diabetes"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}\n(threshold={threshold})")
    plt.tight_layout()

    if save:
        path = f"outputs/figures/cm_{model_name.replace(' ', '_')}.png"
        fig.savefig(path, dpi=150)
        print(f"[Plot]  Confusion matrix saved → {path}")
    plt.close(fig)

    # Text report
    print(f"\n[ClassReport] {model_name}")
    print(classification_report(y_true, y_pred, target_names=["No Diabetes", "Diabetes"]))


# ── 3. ROC CURVE ─────────────────────────────────────────────────────────────

def plot_roc_curves(
    y_true: np.ndarray,
    oof_predictions: dict,
    save: bool = True
) -> None:
    """
    Plot ROC curves for all models on a single figure.

    Args:
        y_true:          ground truth labels
        oof_predictions: dict of {model_name: oof_proba_array}
        save:            write figure to disk
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")

    for name, preds in oof_predictions.items():
        fpr, tpr, _ = roc_curve(y_true, preds)
        score       = roc_auc_score(y_true, preds)
        ax.plot(fpr, tpr, lw=2, label=f"{name}  (AUC = {score:.4f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — All Models (OOF Predictions)", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        path = "outputs/figures/roc_curves.png"
        fig.savefig(path, dpi=150)
        print(f"[Plot]  ROC curves saved → {path}")
    plt.close(fig)


# ── 4. PRECISION-RECALL CURVE ─────────────────────────────────────────────────

def plot_precision_recall_curves(
    y_true: np.ndarray,
    oof_predictions: dict,
    save: bool = True
) -> None:
    """
    Plot Precision-Recall curves for all models.
    Especially useful under class imbalance.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, preds in oof_predictions.items():
        prec, rec, _ = precision_recall_curve(y_true, preds)
        ap           = average_precision_score(y_true, preds)
        ax.plot(rec, prec, lw=2, label=f"{name}  (AP = {ap:.4f})")

    baseline = y_true.mean()
    ax.axhline(y=baseline, color="k", linestyle="--", lw=1,
               label=f"Baseline (AP = {baseline:.2f})")

    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — All Models", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        path = "outputs/figures/pr_curves.png"
        fig.savefig(path, dpi=150)
        print(f"[Plot]  PR curves saved → {path}")
    plt.close(fig)


# ── 5. FEATURE IMPORTANCE PLOT ────────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_names: list,
    model_name: str = "LightGBM",
    top_n: int = 20,
    save: bool = True
) -> None:
    """
    Plot top-N feature importances for tree-based models.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print(f"[Warn]  {model_name} has no feature_importances_ attribute — skipping.")
        return

    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=fi_df, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=13)
    ax.set_xlabel("Importance", fontsize=11)
    plt.tight_layout()

    if save:
        path = f"outputs/figures/feat_imp_{model_name.replace(' ', '_')}.png"
        fig.savefig(path, dpi=150)
        print(f"[Plot]  Feature importance saved → {path}")
    plt.close(fig)


# ── 6. ENSEMBLE: SIMPLE AVERAGING ────────────────────────────────────────────

def ensemble_average(predictions: dict) -> np.ndarray:
    """
    Simple (unweighted) average of all model predictions.

    Args:
        predictions: dict of {model_name: proba_array}

    Returns:
        averaged prediction array
    """
    all_preds = np.column_stack(list(predictions.values()))
    ensemble  = all_preds.mean(axis=1)
    print(f"[Ensemble] Simple average of: {list(predictions.keys())}")
    return ensemble


def ensemble_average_top_k(
    predictions: dict,
    cv_scores:   dict,
    k: int = 3
) -> np.ndarray:
    """
    Average the top-k models by CV score.

    Args:
        predictions: dict of {model_name: proba_array}
        cv_scores:   dict of {model_name: auc_score}
        k:           number of top models to include

    Returns:
        ensemble prediction array
    """
    top_k = sorted(cv_scores, key=cv_scores.get, reverse=True)[:k]
    print(f"[Ensemble] Top-{k} models by CV AUC: {top_k}")
    selected  = {name: predictions[name] for name in top_k if name in predictions}
    all_preds = np.column_stack(list(selected.values()))
    return all_preds.mean(axis=1)


# ── 7. ENSEMBLE: WEIGHTED BLENDING ───────────────────────────────────────────

def ensemble_weighted_blend(
    predictions: dict,
    cv_scores:   dict
) -> np.ndarray:
    """
    Weighted blend: weights proportional to CV AUC scores.

    Args:
        predictions: dict of {model_name: proba_array}
        cv_scores:   dict of {model_name: auc_score}

    Returns:
        weighted blend prediction array
    """
    total_score = sum(cv_scores[name] for name in predictions)
    weights     = {name: cv_scores[name] / total_score for name in predictions}

    print("[Ensemble] Weighted blend:")
    for name, w in weights.items():
        print(f"  {name:<25}  weight = {w:.4f}")

    ensemble = sum(
        weights[name] * predictions[name]
        for name in predictions
    )
    return ensemble


# ── 8. GENERATE KAGGLE SUBMISSION ────────────────────────────────────────────

def generate_submission(
    test_ids: pd.Series,
    predictions: np.ndarray,
    path: str = "outputs/submission.csv"
) -> pd.DataFrame:
    """
    Create a Kaggle-ready submission CSV.

    Format:
        id, diabetes

    Note: predictions must be PROBABILITIES (not labels).

    Args:
        test_ids:    ID column from test set
        predictions: predicted probabilities
        path:        output file path

    Returns:
        submission DataFrame
    """
    submission = pd.DataFrame({
        "id":       test_ids.values,
        "diabetes": predictions
    })

    # Sanity checks
    assert submission["diabetes"].between(0, 1).all(), \
        "Predictions must be probabilities [0, 1]!"
    assert len(submission) == len(test_ids), "Row count mismatch!"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    submission.to_csv(path, index=False)
    print(f"\n[Submission] Saved {len(submission)} rows → {path}")
    print(submission.head(3).to_string(index=False))
    return submission


# ── 9. CV vs LEADERBOARD GAP ANALYSIS ────────────────────────────────────────

def cv_leaderboard_analysis(
    cv_scores:         dict,
    leaderboard_score: float = None,
    best_model:        str   = None
) -> pd.DataFrame:
    """
    Compare CV scores across models. If leaderboard score is known,
    compute the gap (overfitting indicator).

    Args:
        cv_scores:         dict {model_name: cv_auc}
        leaderboard_score: public LB score (optional)
        best_model:        name of selected best model

    Returns:
        Summary DataFrame
    """
    df = pd.DataFrame.from_dict(cv_scores, orient="index", columns=["CV_AUC"])
    df = df.sort_values("CV_AUC", ascending=False)

    if leaderboard_score is not None and best_model in df.index:
        df["LB_AUC"] = np.nan
        df.loc[best_model, "LB_AUC"] = leaderboard_score
        df["Gap"] = df["CV_AUC"] - df["LB_AUC"]
        gap = df.loc[best_model, "Gap"]
        status = "⚠️  OVERFITTING DETECTED" if gap > 0.01 else "✅  Gap within tolerance"
        print(f"\n[Gap Analysis]  CV: {df.loc[best_model, 'CV_AUC']:.5f}  |  "
              f"LB: {leaderboard_score:.5f}  |  Gap: {gap:.5f}  →  {status}")

    print("\n[Model Comparison]")
    print(df.to_string())
    return df


# ── 10. MASTER EVALUATION PIPELINE ───────────────────────────────────────────

def run_evaluation_pipeline(
    y_train:         np.ndarray,
    oof_predictions: dict,
    cv_scores:       dict,
    models:          dict,
    X_test:          np.ndarray,
    X_test_scaled:   np.ndarray,
    test_ids:        pd.Series,
    feature_names:   list,
    leaderboard_score: float = None
) -> dict:
    """
    Master evaluation function:
      1. Evaluate all models (AUC, confusion matrix, reports)
      2. Plot ROC + PR curves
      3. Plot feature importances
      4. Build ensembles
      5. Generate submission CSV
      6. CV vs LB gap analysis

    Returns:
        dict with ensemble predictions and submission path
    """
    print("\n" + "="*60)
    print("  EVALUATION PIPELINE")
    print("="*60)

    # ── Per-model evaluation ──────────────────────────────────────────────
    print("\n[1] Per-model OOF ROC-AUC")
    for name, preds in oof_predictions.items():
        evaluate_roc_auc(y_train, preds, name)
        plot_confusion_matrix(y_train, preds, model_name=name)

    # ── Visualization ─────────────────────────────────────────────────────
    print("\n[2] Plotting curves …")
    plot_roc_curves(y_train, oof_predictions)
    plot_precision_recall_curves(y_train, oof_predictions)

    # ── Feature importances ───────────────────────────────────────────────
    print("\n[3] Feature importances …")
    tree_models = ["lightgbm", "xgboost", "catboost", "random_forest"]
    for name in tree_models:
        if name in models:
            plot_feature_importance(models[name], feature_names, model_name=name)

    # ── Ensemble OOF evaluation ───────────────────────────────────────────
    print("\n[4] Ensemble evaluation …")
    avg_oof = ensemble_average(oof_predictions)
    avg_auc = roc_auc_score(y_train, avg_oof)
    print(f"[Ensemble] Simple avg OOF AUC : {avg_auc:.5f}")

    top3_oof = ensemble_average_top_k(oof_predictions, cv_scores, k=3)
    top3_auc = roc_auc_score(y_train, top3_oof)
    print(f"[Ensemble] Top-3 avg  OOF AUC : {top3_auc:.5f}")

    wt_oof   = ensemble_weighted_blend(oof_predictions, cv_scores)
    wt_auc   = roc_auc_score(y_train, wt_oof)
    print(f"[Ensemble] Weighted   OOF AUC : {wt_auc:.5f}")

    # ── Test set predictions ──────────────────────────────────────────────
    print("\n[5] Generating test predictions …")
    test_preds_all = {}
    scaled_models  = {"logistic_regression"}      # models requiring scaled input

    for name, model in models.items():
        X_input = X_test_scaled if name in scaled_models else X_test
        test_preds_all[name] = model.predict_proba(X_input)[:, 1]

    # Best single model (by CV AUC)
    best_name    = max(cv_scores, key=cv_scores.get)
    final_preds  = test_preds_all[best_name]
    print(f"[Submit] Using best single model: {best_name}  (CV AUC={cv_scores[best_name]:.5f})")

    # Also prepare ensemble submission
    ensemble_test_preds = ensemble_weighted_blend(test_preds_all, cv_scores)

    # Generate both submission files
    sub_single   = generate_submission(test_ids, final_preds,
                                       path="outputs/submission_best_model.csv")
    sub_ensemble = generate_submission(test_ids, ensemble_test_preds,
                                       path="outputs/submission_ensemble.csv")

    # ── Gap analysis ──────────────────────────────────────────────────────
    print("\n[6] CV vs Leaderboard gap analysis …")
    extended_cv = {**cv_scores,
                   "ensemble_avg":     avg_auc,
                   "ensemble_top3":    top3_auc,
                   "ensemble_weighted": wt_auc}
    gap_df = cv_leaderboard_analysis(extended_cv, leaderboard_score, best_name)
    gap_df.to_csv("outputs/model_comparison.csv")
    print("[Save]  Model comparison saved → outputs/model_comparison.csv")

    return {
        "best_model":             best_name,
        "test_preds_best":        final_preds,
        "test_preds_ensemble":    ensemble_test_preds,
        "submission_single_path": "outputs/submission_best_model.csv",
        "submission_ensemble_path": "outputs/submission_ensemble.csv",
        "gap_df":                 gap_df,
    }


# ── EXAMPLE USAGE ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from preprocessing  import run_preprocessing_pipeline
    from model_training import run_training_pipeline

    # Step 1 — Preprocessing
    data = run_preprocessing_pipeline(
        train_path="data/train.csv",
        test_path="data/test.csv"
    )

    # Step 2 — Training
    results = run_training_pipeline(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_train_scaled=data["X_train_scaled"],
        tune=False
    )

    # Step 3 — Evaluation + Submission
    eval_results = run_evaluation_pipeline(
        y_train=data["y_train"],
        oof_predictions=results["oof_predictions"],
        cv_scores=results["cv_scores"],
        models=results["models"],
        X_test=data["X_test"],
        X_test_scaled=data["X_test_scaled"],
        test_ids=data["test_ids"],
        feature_names=data["feature_names"],
        leaderboard_score=None   # Update after first Kaggle submission
    )

    print("\n[evaluation.py] ✅ Done!")
    print(f"  Best model      : {eval_results['best_model']}")
    print(f"  Submission (single)  : {eval_results['submission_single_path']}")
    print(f"  Submission (ensemble): {eval_results['submission_ensemble_path']}")
