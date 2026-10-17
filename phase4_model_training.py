"""
Phase 4: Regression Model Training & Cross-Validation
Internship: CodeAlpha Data Science Internship - Task 3: Car Price Prediction

This module implements Phase 4 only:
  1. Loads and defensively validates the Phase 3 modeling dataset ('cleaned_car_data.csv').
  2. Partitions data once into training (80%) and holdout test (20%) partitions (random_state=42).
  3. Formulates a scikit-learn Pipeline with ColumnTransformer (StandardScaler + OneHotEncoder).
  4. Evaluates 5 candidate regression algorithms using 5-fold cross-validation on the training set only:
       - Linear Regression
       - Ridge Regression
       - Decision Tree Regressor
       - Random Forest Regressor
       - Gradient Boosting Regressor
  5. Computes mean and standard deviation for CV RMSE, MAE, and R^2.
  6. Selects the champion candidate with the lowest mean CV RMSE.
  7. Refits the champion pipeline on the complete training set only (holdout test set untouched).
  8. Serializes best_model.joblib, cross_validation_results.csv, and training_metadata.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# -------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -------------------------------------------------------------------------

TARGET: str = "Selling_Price"

NUMERIC_FEATURES: List[str] = [
    "Present_Price",
    "Driven_kms",
    "Car_Age",
    "Owner",
]

CATEGORICAL_FEATURES: List[str] = [
    "Fuel_Type",
    "Selling_type",
    "Transmission",
]

FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
N_SPLITS: int = 5


def print_section_header(title: str) -> None:
    """Print a clean visual section header for terminal output."""
    separator = "=" * 72
    print(f"\n{separator}")
    print(f" {title.upper()}")
    print(separator)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for Phase 4 execution."""
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Phase 4: Car Price Prediction - Regression Model Training & Cross-Validation"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=base / "outputs" / "phase3" / "cleaned_car_data.csv",
        help="Path to Phase 3 cleaned modeling dataset (defaults to 'outputs/phase3/cleaned_car_data.csv')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "outputs" / "phase4",
        help="Directory to save Phase 4 artifacts (defaults to 'outputs/phase4')",
    )
    return parser.parse_args()


def load_and_validate(path: Path) -> pd.DataFrame:
    """
    Load and defensively validate Phase 3 cleaned dataset.
    Verifies schema, positive target values, zero nulls, zero infs, and no duplicates.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"[FILE ERROR] Phase 3 cleaned dataset not found at: {path.resolve()}.\n"
            "Action Required: Execute Phase 3 preprocessing first."
        )

    df = pd.read_csv(path)

    if df.columns.duplicated().any():
        dup_names = df.columns[df.columns.duplicated()].unique().tolist()
        raise ValueError(f"[SCHEMA ERROR] Duplicate column names found in Phase 3 dataset: {dup_names}")

    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"[SCHEMA ERROR] Missing required modeling column(s): {missing}")

    # Check for missing values
    if df[required].isna().any().any():
        counts = df[required].isna().sum()
        counts = counts[counts > 0].to_dict()
        raise ValueError(f"[DATA QUALITY ERROR] Missing values detected in modeling columns: {counts}")

    # Check for finite numeric values
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        values = df[col].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"[DATA QUALITY ERROR] Non-finite (+/- Inf or NaN) values detected in column '{col}'.")

    # Target must be strictly positive
    if (df[TARGET] <= 0).any():
        non_pos_count = int((df[TARGET] <= 0).sum())
        raise ValueError(f"[TARGET ERROR] Target '{TARGET}' must be strictly positive (found {non_pos_count} non-positive value(s)).")

    # Invariant: No duplicate rows in cleaned modeling data
    if df.duplicated().any():
        dup_count = int(df.duplicated().sum())
        raise ValueError(
            f"[DATA QUALITY ERROR] {dup_count} duplicate row(s) found in Phase 3 dataset. "
            "Please review Phase 3 deduplication before model training."
        )

    # Clean categorical text columns
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").str.strip()
        if (df[col].isna() | (df[col].str.len() == 0)).any():
            raise ValueError(f"[DATA QUALITY ERROR] Blank or missing category values found in column '{col}'.")

    if len(df) < N_SPLITS * 2:
        raise ValueError(
            f"[DATA SIZE ERROR] Dataset has {len(df)} records. At least {N_SPLITS * 2} records "
            "are required for five-fold cross-validation."
        )

    return df


def make_one_hot_encoder() -> OneHotEncoder:
    """
    Construct OneHotEncoder with cross-version scikit-learn compatibility.
    Supports both modern 'sparse_output' and legacy 'sparse' keyword arguments.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipeline(estimator: Any) -> Pipeline:
    """
    Assemble ColumnTransformer preprocessor and regression estimator inside a Pipeline.
    Prevents data leakage: scalers and encoders are fitted strictly within cross-validation folds.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                make_one_hot_encoder(),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def get_models() -> Dict[str, Any]:
    """
    Return dictionary of 5 benchmark regression algorithms configured with fixed random seeds.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(
            random_state=RANDOM_STATE,
            min_samples_leaf=2,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
    }


def run_phase4(csv_path: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Main orchestration routine for Phase 4 Model Training and Cross-Validation.
    """
    print_section_header("CodeAlpha Data Science Internship - Task 3: Car Price Prediction")
    print(" PHASE 4: REGRESSION MODEL TRAINING & CROSS-VALIDATION")
    print("=" * 72)
    print(f"Phase 3 Dataset Path : {csv_path.resolve()}")
    print(f"Output Directory     : {output_dir.resolve()}")

    df = load_and_validate(csv_path)

    # Segregate features and target
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Sanity Invariant Check 1: Target cannot be among predictors
    if TARGET in X.columns:
        raise AssertionError(f"[LEAKAGE ERROR] Target variable '{TARGET}' detected among input predictors.")

    # Train-test split (80% train, 20% reserved test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Sanity Invariant Check 2: X_test and y_test are strictly reserved
    print(f"\nDataset Dimensions   : {len(df)} total rows")
    print(f"Training Partition   : {len(X_train)} rows (80.0%) - Used for CV & Model Selection")
    print(f"Reserved Test Set    : {len(X_test)} rows (20.0%) - Strictly held out for Phase 5 Evaluation")
    print(f"Predictor Features   : {FEATURES}")
    print(f"Target Column        : '{TARGET}'")

    # 5-fold shuffled cross-validation on training partition only
    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    results: List[Dict[str, Any]] = []
    candidates = get_models()

    print_section_header("Cross-Validation Benchmarking (Training Partition Only)")
    print("  * Metric Guide:")
    print("    - RMSE (Root Mean Squared Error): Penalizes large residual errors. Lower is better.")
    print("    - MAE (Mean Absolute Error): Average magnitude of price errors. Lower is better.")
    print("    - R^2 (Coefficient of Determination): Proportion of target variance explained. Higher is better.")
    print("      (Note: R^2 indicates relative variance explained, NOT a percentage of prediction accuracy).\n")

    for model_name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)

        cv_result = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1,
            error_score="raise",
        )

        # Scikit-learn negates errors so that higher score = better
        rmse_values = -cv_result["test_rmse"]
        mae_values = -cv_result["test_mae"]
        r2_values = cv_result["test_r2"]

        row = {
            "model": model_name,
            "cv_rmse_mean": float(rmse_values.mean()),
            "cv_rmse_std": float(rmse_values.std(ddof=1)),
            "cv_mae_mean": float(mae_values.mean()),
            "cv_mae_std": float(mae_values.std(ddof=1)),
            "cv_r2_mean": float(r2_values.mean()),
            "cv_r2_std": float(r2_values.std(ddof=1)),
        }
        results.append(row)

        print(
            f"  {model_name:20s} | "
            f"RMSE: {row['cv_rmse_mean']:.4f} (± {row['cv_rmse_std']:.4f}) | "
            f"MAE: {row['cv_mae_mean']:.4f} (± {row['cv_mae_std']:.4f}) | "
            f"R²: {row['cv_r2_mean']:.4f} (± {row['cv_r2_std']:.4f})"
        )

    # Sort candidates: primary rank by lowest RMSE mean, tie-breaker lowest MAE mean
    results_df = pd.DataFrame(results).sort_values(
        by=["cv_rmse_mean", "cv_mae_mean", "model"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    selected_name = str(results_df.iloc[0]["model"])
    selected_estimator = candidates[selected_name]

    print(f"\n[SELECTION DECISION] Champion candidate selected: '{selected_name}'")
    print(f"  - Lowest Mean CV RMSE: {results_df.iloc[0]['cv_rmse_mean']:.4f} dataset price units")
    print(f"  - Mean CV MAE        : {results_df.iloc[0]['cv_mae_mean']:.4f} dataset price units")
    print(f"  - Mean CV R²         : {results_df.iloc[0]['cv_r2_mean']:.4f}")

    # Refit the champion pipeline on the FULL training partition only
    print(f"\nRefitting '{selected_name}' pipeline on complete training partition (n={len(X_train)})...")
    best_pipeline = make_pipeline(selected_estimator)
    best_pipeline.fit(X_train, y_train)

    # Save artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_model.joblib"
    results_path = output_dir / "cross_validation_results.csv"
    metadata_path = output_dir / "training_metadata.json"

    joblib.dump(best_pipeline, model_path)
    results_df.to_csv(results_path, index=False)

    # Sanity Invariant Check 3: Verify the saved model can be loaded and predict
    loaded_model = joblib.load(model_path)
    sample_preds = loaded_model.predict(X_train.iloc[:3])
    if not np.isfinite(sample_preds).all() or len(sample_preds) != 3:
        raise AssertionError("[SERIALIZATION ERROR] Reloaded pipeline failed to generate valid predictions.")

    # Read reference year from Phase 3 metadata if present
    phase3_summary_path = csv_path.parent / "preprocessing_summary.json"
    reference_year = None
    if phase3_summary_path.is_file():
        try:
            with phase3_summary_path.open(encoding="utf-8") as f:
                phase3_doc = json.load(f)
            reference_year = phase3_doc.get("feature_engineering", {}).get("reference_year")
        except Exception:
            reference_year = None

    metadata = {
        "phase": 4,
        "target": TARGET,
        "features": FEATURES,
        "numerical_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "training_rows": len(X_train),
        "reserved_test_rows": len(X_test),
        "cross_validation": {
            "strategy": "KFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "data_used": "Training partition only (test set held out)",
        },
        "selection_metric": "Lowest mean cross-validation RMSE",
        "selected_model": selected_name,
        "reference_year": reference_year,
        "test_partition_used_for_model_selection": False,
        "test_partition_evaluated_in_phase4": False,
        "scikit_learn_version": sklearn.__version__,
        "champion_cv_metrics": {
            "cv_rmse_mean": float(results_df.iloc[0]["cv_rmse_mean"]),
            "cv_rmse_std": float(results_df.iloc[0]["cv_rmse_std"]),
            "cv_mae_mean": float(results_df.iloc[0]["cv_mae_mean"]),
            "cv_mae_std": float(results_df.iloc[0]["cv_mae_std"]),
            "cv_r2_mean": float(results_df.iloc[0]["cv_r2_mean"]),
            "cv_r2_std": float(results_df.iloc[0]["cv_r2_std"]),
        },
        "artifacts": {
            "model": str(model_path.resolve()),
            "cross_validation_results": str(results_path.resolve()),
            "metadata": str(metadata_path.resolve()),
        },
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print_section_header("Phase 4 Execution Summary")
    print(f"Champion Model Serialized: {model_path.resolve()}")
    print(f"CV Benchmarks Table      : {results_path.resolve()}")
    print(f"Training Metadata        : {metadata_path.resolve()}")
    print("=" * 72)
    print(" PHASE 4 REGRESSION MODEL TRAINING CONCLUDED SUCCESSFULLY")
    print("=" * 72 + "\n")

    return {
        "best_pipeline": best_pipeline,
        "results_df": results_df,
        "selected_model": selected_name,
        "metadata": metadata,
    }


def main() -> None:
    args = parse_args()
    results = run_phase4(args.csv, args.output_dir)
    if not results:
        sys.exit(1)


if __name__ == "__main__":
    main()
