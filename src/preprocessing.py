"""
=============================================================================
MEMBER 1 — DATA ENGINEER: preprocessing.py
=============================================================================
Responsibilities:
  - Load and explore dataset
  - Handle missing values
  - Detect class imbalance
  - Perform feature engineering
  - Encode categorical variables
  - Normalize / scale numerical features
  - Output clean dataset for modeling
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ── 1. DATA LOADING ──────────────────────────────────────────────────────────

def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test CSV files.

    Args:
        train_path: path to train.csv
        test_path:  path to test.csv

    Returns:
        (train_df, test_df) as pandas DataFrames
    """
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    print(f"[Load]  Train shape : {train_df.shape}")
    print(f"[Load]  Test  shape : {test_df.shape}")
    return train_df, test_df


# ── 2. BASIC EDA SUMMARY ─────────────────────────────────────────────────────

def basic_eda(df: pd.DataFrame, label: str = "Dataset") -> None:
    """
    Print a concise EDA summary: dtypes, nulls, target distribution.
    """
    print(f"\n{'='*60}")
    print(f"  EDA — {label}")
    print(f"{'='*60}")
    print(f"\n[Shape]  {df.shape}")
    print("\n[dtypes]\n", df.dtypes.value_counts())
    print("\n[Missing values]\n", df.isnull().sum()[df.isnull().sum() > 0])

    if "diagnosed_diabetes" in df.columns:
        vc = df["diagnosed_diabetes"].value_counts(normalize=True) * 100
        print(f"\n[Target distribution]\n{vc.round(2)}")
        imbalance_ratio = vc.iloc[0] / vc.iloc[1]
        print(f"\n[Class imbalance ratio]  {imbalance_ratio:.2f}:1")


# ── 3. MISSING VALUE HANDLING ────────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
      - Numerical  → median  (robust to outliers)
      - Categorical → mode
    """
    df = df.copy()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Remove id/target from imputation targets
    for c in ["id", "diagnosed_diabetes"]:
        if c in num_cols:
            num_cols.remove(c)

    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
        print(f"[Impute] Numerical  ({len(num_cols)} cols) → median")

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
        print(f"[Impute] Categorical({len(cat_cols)} cols) → mode")

    return df


# ── 4. FEATURE ENGINEERING (corrected for actual dataset) ────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create meaningful interaction features for diabetes prediction.
    Based on actual columns: bmi, age, systolic_bp, diastolic_bp,
    cholesterol_total, hdl_cholesterol, ldl_cholesterol, etc.

    New features:
      - bmi_age_interaction   : BMI × age
      - bp_cholesterol_ratio  : (systolic_bp + diastolic_bp) / (cholesterol_total + 1)
      - bp_bmi_ratio          : (systolic_bp + diastolic_bp) / (bmi + 1)
      - high_bp_flag          : 1 if systolic_bp ≥ 130 or diastolic_bp ≥ 80
      - obese_flag            : 1 if bmi ≥ 30
      - age_group             : bucketed age category
      - cholesterol_ratio     : total_cholesterol / (hdl_cholesterol + 1)
    """
    df = df.copy()
    cols = df.columns.tolist()

    # 1. BMI × age
    if "bmi" in cols and "age" in cols:
        df["bmi_age_interaction"] = df["bmi"] * df["age"]

    # 2. Blood pressure × cholesterol ratio (cardiovascular-metabolic link)
    if all(c in cols for c in ["systolic_bp", "diastolic_bp", "cholesterol_total"]):
        df["bp_cholesterol_ratio"] = (df["systolic_bp"] + df["diastolic_bp"]) / (df["cholesterol_total"] + 1)

    # 3. Blood pressure / BMI ratio
    if all(c in cols for c in ["systolic_bp", "diastolic_bp", "bmi"]):
        df["bp_bmi_ratio"] = (df["systolic_bp"] + df["diastolic_bp"]) / (df["bmi"] + 1)

    # 4. High blood pressure flag (hypertension threshold)
    if all(c in cols for c in ["systolic_bp", "diastolic_bp"]):
        df["high_bp_flag"] = ((df["systolic_bp"] >= 130) | (df["diastolic_bp"] >= 80)).astype(int)

    # 5. Obese flag
    if "bmi" in cols:
        df["obese_flag"] = (df["bmi"] >= 30).astype(int)

    # 6. Age group (non-linear)
    if "age" in cols:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 45, 55, 65, 120],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(int)

    # 7. Cholesterol ratio (total / HDL)
    if all(c in cols for c in ["cholesterol_total", "hdl_cholesterol"]):
        df["cholesterol_ratio"] = df["cholesterol_total"] / (df["hdl_cholesterol"] + 1)

    print(f"[FeatEng] Dataset now has {df.shape[1]} columns")
    return df
  
# ── 5. CATEGORICAL ENCODING ──────────────────────────────────────────────────

def encode_categoricals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode categorical columns:
      - Binary (2 unique) → LabelEncoder
      - Multi-class       → One-hot encoding
    No data leakage: fit only on train.
    """
    cat_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Remove target if accidentally in there
    if "diagnosed_diabetes" in cat_cols:
        cat_cols.remove("diagnosed_diabetes")

    le_dict   = {}
    ohe_cols  = []

    for col in cat_cols:
        n_unique = train_df[col].nunique()
        if n_unique <= 2:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col].astype(str))
            test_df[col]  = le.transform(test_df[col].astype(str))
            le_dict[col]  = le
            print(f"[Encode] LabelEncoder  → {col}")
        else:
            ohe_cols.append(col)

    if ohe_cols:
        train_df = pd.get_dummies(train_df, columns=ohe_cols, drop_first=True)
        test_df  = pd.get_dummies(test_df,  columns=ohe_cols, drop_first=True)
        # Align columns (test may miss some dummy columns after split)
        test_df  = test_df.reindex(columns=train_df.columns, fill_value=0)
        print(f"[Encode] OneHotEncoding → {ohe_cols}")

    return train_df, test_df


# ── 6. FEATURE SCALING ───────────────────────────────────────────────────────

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_path: str = "models/scaler.pkl"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit StandardScaler on train, transform both train and test.
    Saves the fitted scaler for later inference.
    No data leakage: scaler fitted only on X_train.

    Args:
        X_train:     training feature matrix
        X_test:      test feature matrix
        scaler_path: where to persist the fitted scaler

    Returns:
        (X_train_scaled, X_test_scaled) as numpy arrays
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[Scale]  StandardScaler fitted and saved → {scaler_path}")

    return X_train_scaled, X_test_scaled


# ── 7. FULL PREPROCESSING PIPELINE ───────────────────────────────────────────

def run_preprocessing_pipeline(
    train_path: str = "data/train.csv",
    test_path:  str = "data/test.csv",
    output_dir: str = "data/"
) -> dict:
    """
    Master function: runs the complete preprocessing pipeline.

    Returns a dict with keys:
        X_train, y_train, X_test, test_ids,
        X_train_scaled, X_test_scaled, feature_names
    """
    # 1. Load
    train_df, test_df = load_data(train_path, test_path)

    # 2. EDA summary
    basic_eda(train_df, "Train")
    basic_eda(test_df,  "Test")

    # 3. Separate target & IDs
    y_train  = train_df["diagnosed_diabetes"].copy()
    y_train = y_train.astype(int)
    test_ids = test_df["id"].copy()

    train_df = train_df.drop(columns=["diagnosed_diabetes"])
    # Drop 'id' — not predictive
    train_df = train_df.drop(columns=["id"], errors="ignore")
    test_df  = test_df.drop(columns=["id"],  errors="ignore")

    # 4. Missing values
    train_df = handle_missing_values(train_df)
    test_df  = handle_missing_values(test_df)

    # 5. Feature engineering
    train_df = engineer_features(train_df)
    test_df  = engineer_features(test_df)

    # 6. Encode categoricals
    train_df, test_df = encode_categoricals(train_df, test_df)

    # 7. Scale
    feature_names      = train_df.columns.tolist()
    X_train_scaled, X_test_scaled = scale_features(
        train_df, test_df, scaler_path="models/scaler.pkl"
    )

    # 8. Save clean CSVs (unscaled, for tree-based models)
    os.makedirs(output_dir, exist_ok=True)
    train_df["diagnosed_diabetes"] = y_train.values
    train_df.to_csv(f"{output_dir}/train_clean.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_clean.csv",   index=False)
    print(f"\n[Save]  Clean data saved to {output_dir}")

    print(f"\n[Done]  X_train: {X_train_scaled.shape}  |  y_train: {y_train.shape}")
    print(f"[Done]  X_test : {X_test_scaled.shape}")

    return {
        "X_train":        train_df.drop(columns=["diagnosed_diabetes"]).values,
        "y_train":        y_train.values,
        "X_test":         test_df.values,
        "test_ids":       test_ids,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled":  X_test_scaled,
        "feature_names":  feature_names,
    }


# ── EXAMPLE USAGE ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = run_preprocessing_pipeline(
        train_path="data/train.csv",
        test_path="data/test.csv",
        output_dir="data/"
    )
    print("\n[preprocessing.py] Pipeline complete. Keys:", list(data.keys()))
