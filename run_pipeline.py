"""
=============================================================================
run_pipeline.py — MASTER PIPELINE ORCHESTRATOR
=============================================================================
Run this single script to execute the entire ML pipeline:
    1. Preprocessing (Member 1)
    2. Model Training (Member 2)
    3. Evaluation + Submission (Member 3)

Usage:
    python run_pipeline.py
    python run_pipeline.py --tune          # with Optuna tuning
    python run_pipeline.py --lb 0.8712     # with leaderboard score
=============================================================================
"""

import sys
import argparse
import os

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing  import run_preprocessing_pipeline
from model_training import run_training_pipeline
from evaluation     import run_evaluation_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Diabetes Prediction Pipeline")
    parser.add_argument("--train",    default="data/train.csv",  help="Path to train.csv")
    parser.add_argument("--test",     default="data/test.csv",   help="Path to test.csv")
    parser.add_argument("--tune",     action="store_true",       help="Run Optuna tuning")
    parser.add_argument("--n_trials", type=int, default=50,      help="Optuna trial count")
    parser.add_argument("--lb",       type=float, default=None,  help="Leaderboard AUC score")
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*65)
    print("  🩺 DIABETES PREDICTION — FULL ML PIPELINE")
    print("="*65)

    # ── STEP 1: PREPROCESSING ─────────────────────────────────────────────
    print("\n▶ STEP 1/3 — Preprocessing …")
    data = run_preprocessing_pipeline(
        train_path=args.train,
        test_path=args.test,
        output_dir="data/"
    )

    # ── STEP 2: MODEL TRAINING ────────────────────────────────────────────
    print("\n▶ STEP 2/3 — Model Training …")
    results = run_training_pipeline(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_train_scaled=data["X_train_scaled"],
        tune=args.tune,
        n_trials=args.n_trials
    )

    # ── STEP 3: EVALUATION + SUBMISSION ──────────────────────────────────
    print("\n▶ STEP 3/3 — Evaluation + Submission …")
    eval_results = run_evaluation_pipeline(
        y_train=data["y_train"],
        oof_predictions=results["oof_predictions"],
        cv_scores=results["cv_scores"],
        models=results["models"],
        X_test=data["X_test"],
        X_test_scaled=data["X_test_scaled"],
        test_ids=data["test_ids"],
        feature_names=data["feature_names"],
        leaderboard_score=args.lb
    )

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  ✅ PIPELINE COMPLETE — SUMMARY")
    print("="*65)
    print(f"  Best model         : {eval_results['best_model']}")
    print(f"  Submission (single): {eval_results['submission_single_path']}")
    print(f"  Submission (ensemble): {eval_results['submission_ensemble_path']}")
    print(f"\n  📂 All outputs in:   outputs/")
    print(f"  📊 Figures in:        outputs/figures/")
    print(f"  🤖 Models in:         models/")
    print("="*65)


if __name__ == "__main__":
    main()
