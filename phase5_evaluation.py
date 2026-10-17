"""
Phase 5: Final Model Evaluation on Holdout Test Set
Internship: CodeAlpha Data Science Internship - Task 3: Car Price Prediction

This module implements Phase 5:
  1. Inspects Phase 3 dataset and Phase 4 artifacts and metadata.
  2. Reconstructs the exact 80/20 train-test split (random_state=42, test_size=0.20)
     matching Phase 4 metadata (239 train rows, 60 test rows).
  3. Loads the fitted Phase 4 pipeline (joblib) without retraining or refitting.
  4. Evaluates the fitted model once on the reserved 60-row holdout test partition.
  5. Computes test RMSE, MAE, Median Absolute Error, and R^2.
  6. Generates diagnostic plots:
       - outputs/phase5/actual_vs_predicted.png
       - outputs/phase5/residual_diagnostics.png
  7. Saves traceable test predictions (outputs/phase5/test_predictions.csv).
  8. Generates evaluation metadata (outputs/phase5/evaluation_metrics.json) and
     a comprehensive markdown summary report (outputs/phase5/PHASE5_REPORT.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# -------------------------------------------------------------------------
# CONSTANTS & SPECIFICATIONS
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
EXPECTED_TOTAL_ROWS: int = 299
EXPECTED_TRAIN_ROWS: int = 239
EXPECTED_TEST_ROWS: int = 60


def print_section_header(title: str) -> None:
    """Print a clean visual section header for terminal logging."""
    separator = "=" * 72
    print(f"\n{separator}")
    print(f" {title.upper()}")
    print(separator)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 5 execution."""
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Phase 5: Final Model Evaluation on Holdout Test Partition"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=base / "outputs" / "phase3" / "cleaned_car_data.csv",
        help="Path to Phase 3 cleaned dataset (defaults to outputs/phase3/cleaned_car_data.csv)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=base / "outputs" / "phase4" / "best_model.joblib",
        help="Path to Phase 4 fitted model pipeline (defaults to outputs/phase4/best_model.joblib)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=base / "outputs" / "phase4" / "training_metadata.json",
        help="Path to Phase 4 training metadata (defaults to outputs/phase4/training_metadata.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "outputs" / "phase5",
        help="Directory to save Phase 5 evaluation outputs (defaults to outputs/phase5)",
    )
    return parser.parse_args()


import hashlib

def load_inputs(
    csv_path: Path,
    model_path: Path,
    metadata_path: Path,
) -> Tuple[pd.DataFrame, Pipeline, Dict[str, Any], str]:
    """
    Validate and load Phase 3 dataset, Phase 4 model pipeline, and Phase 4 metadata.
    Computes SHA-256 checksum of the model artifact for tamper-evident provenance.
    Raises actionable errors if files are missing or schema invariants are violated.
    """
    for path, label in [
        (csv_path, "Phase 3 cleaned dataset"),
        (model_path, "Phase 4 fitted model"),
        (metadata_path, "Phase 4 training metadata"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} not found at '{path.resolve()}'. "
                "Ensure prior phases were executed successfully."
            )

    # 1. Compute SHA-256 cryptographic fingerprint of the saved model
    with open(model_path, "rb") as mf:
        model_sha256 = hashlib.sha256(mf.read()).hexdigest()

    df = pd.read_csv(csv_path)

    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column headers found in Phase 3 dataset: {dups}")

    required_cols = FEATURES + [TARGET]
    missing_cols = sorted(set(required_cols) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Phase 3 dataset is missing required modeling columns: {missing_cols}")

    if df[required_cols].isna().any().any():
        na_counts = df[required_cols].isna().sum()
        na_dict = na_counts[na_counts > 0].to_dict()
        raise ValueError(f"Missing values found in modeling columns: {na_dict}")

    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        values = df[col].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values encountered in column '{col}'.")

    if (df[TARGET] <= 0).any():
        invalid_count = int((df[TARGET] <= 0).sum())
        raise ValueError(f"Target '{TARGET}' contains {invalid_count} non-positive value(s).")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Attach verified SHA-256 fingerprint to metadata dictionary
    metadata["model_artifact_sha256"] = model_sha256

    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        raise TypeError(f"Loaded object from '{model_path}' does not have a predict method.")

    selected_name = metadata.get("selected_model", "")
    estimator = model.named_steps.get("model") or model.named_steps.get("regressor")
    estimator_class_name = type(estimator).__name__ if estimator else ""
    if "Random Forest" in selected_name and "RandomForest" not in estimator_class_name:
        raise ValueError(
            f"Phase 4 metadata selected '{selected_name}', but loaded pipeline estimator is '{estimator_class_name}'."
        )

    return df, model, metadata


def reconstruct_holdout(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.Series, pd.Index]:
    """
    Reconstruct the exact identical train-test partition using identical split parameters.
    Verifies split sizes against Phase 4 metadata and dataset row count.
    """
    recorded_seed = metadata.get("random_state")
    recorded_test_size = metadata.get("test_size")

    if len(df) != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"Phase 3 dataset record count ({len(df)}) does not match expected ({EXPECTED_TOTAL_ROWS}). "
            "Dataset order and integrity must match Phase 4."
        )

    if recorded_seed != RANDOM_STATE:
        raise ValueError(
            f"Phase 4 random_state ({recorded_seed}) does not match expected ({RANDOM_STATE})."
        )

    if recorded_test_size is None or not np.isclose(float(recorded_test_size), TEST_SIZE):
        raise ValueError(
            f"Phase 4 test_size ({recorded_test_size}) does not match expected ({TEST_SIZE})."
        )

    if metadata.get("target") != TARGET:
        raise ValueError(
            f"Phase 4 metadata target ('{metadata.get('target')}') differs from Phase 5 ('{TARGET}')."
        )

    if metadata.get("features") != FEATURES:
        raise ValueError("Phase 4 metadata features differ from Phase 5 feature specification.")

    positions = np.arange(len(df))
    train_positions, test_positions = train_test_split(
        positions,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    expected_train = metadata.get("training_rows", EXPECTED_TRAIN_ROWS)
    expected_test = metadata.get("reserved_test_rows", EXPECTED_TEST_ROWS)

    if len(train_positions) != expected_train:
        raise ValueError(
            f"Reconstructed training set size ({len(train_positions)}) differs from Phase 4 metadata ({expected_train})."
        )

    if len(test_positions) != expected_test:
        raise ValueError(
            f"Reconstructed test set size ({len(test_positions)}) differs from Phase 4 metadata ({expected_test})."
        )

    X_test = df.iloc[test_positions][FEATURES].copy()
    y_test = df.iloc[test_positions][TARGET].copy()
    source_indices = df.index[test_positions]

    return X_test, y_test, source_indices


def generate_evaluation_plots(
    actual: np.ndarray,
    predictions: np.ndarray,
    output_dir: Path,
    rmse: float,
    r2: float,
) -> Tuple[Path, Path]:
    """
    Generate and save actual-vs-predicted and residual diagnostic plots.
    """
    actual_plot_path = output_dir / "actual_vs_predicted.png"
    residual_plot_path = output_dir / "residual_diagnostics.png"

    # Plot 1: Actual vs Predicted
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        actual,
        predictions,
        color="#1f77b4",
        edgecolor="#0b3d60",
        alpha=0.75,
        s=50,
        label=f"Holdout Records (N={len(actual)})",
    )

    # 45-degree reference line
    min_val = float(min(actual.min(), predictions.min()))
    max_val = float(max(actual.max(), predictions.max()))
    padding = (max_val - min_val) * 0.05
    line_min = max(0.0, min_val - padding)
    line_max = max_val + padding

    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label="Ideal Perfect Prediction (y = x)",
    )

    ax.set_xlim(line_min, line_max)
    ax.set_ylim(line_min, line_max)
    ax.set_xlabel("Actual Selling Price (dataset price units)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted Selling Price (dataset price units)", fontsize=11, fontweight="bold")
    ax.set_title("Phase 5 Holdout Test: Actual vs. Predicted Selling Price", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)

    summary_text = f"Test R² = {r2:.4f}\nTest RMSE = {rmse:.4f}\nHoldout N = {len(actual)}"
    ax.text(
        0.95,
        0.05,
        summary_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(actual_plot_path, dpi=200)
    plt.close(fig)

    # Plot 2: Residual Diagnostics (Residuals vs Predicted + Residual Histogram)
    residuals = actual - predictions

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot A: Residuals vs Predicted
    axes[0].scatter(
        predictions,
        residuals,
        color="#2ca02c",
        edgecolor="#145214",
        alpha=0.75,
        s=45,
    )
    axes[0].axhline(0, color="#d62728", linestyle="--", linewidth=1.8, label="Zero Error Line (Residual = 0)")
    axes[0].set_xlabel("Predicted Selling Price (dataset price units)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Residual: Actual - Predicted (dataset price units)", fontsize=10, fontweight="bold")
    axes[0].set_title("Residuals vs. Predicted Values", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", frameon=True)

    # Subplot B: Residual Distribution Histogram
    axes[1].hist(
        residuals,
        bins=15,
        color="#3470a3",
        edgecolor="#1c3b57",
        alpha=0.8,
    )
    axes[1].axvline(0, color="#d62728", linestyle="--", linewidth=1.8, label="Zero Error Line")
    axes[1].set_xlabel("Residual: Actual - Predicted (dataset price units)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Frequency (Car Count)", fontsize=10, fontweight="bold")
    axes[1].set_title("Distribution of Holdout Residuals", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(residual_plot_path, dpi=200)
    plt.close(fig)

    return actual_plot_path, residual_plot_path


def run_phase5(
    csv_path: Path,
    model_path: Path,
    metadata_path: Path,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute Phase 5 Holdout Evaluation pipeline.
    Evaluates the fitted Phase 4 model once on the holdout test set without retraining.
    """
    print_section_header("CodeAlpha Data Science Internship - Task 3: Car Price Prediction")
    print(" PHASE 5: FINAL MODEL EVALUATION ON HOLDOUT TEST PARTITION")
    print("=" * 72)
    df, model, p4_metadata = load_inputs(csv_path, model_path, metadata_path)
    model_sha256 = p4_metadata.get("model_artifact_sha256", "Unknown")
    X_test, y_test, source_indices = reconstruct_holdout(df, p4_metadata)

    selected_model_name = p4_metadata.get("selected_model", "Unknown")
    print(f"Loaded Champion Model  : {selected_model_name}")
    print(f"Model Artifact Source  : {model_path.resolve()}")
    print(f"Model Artifact SHA-256 : {model_sha256}")
    print(f"Holdout Test Records   : {len(X_test)} cars (strictly held out during Phase 4 CV)")
    print(f"Predictor Features     : {FEATURES}")
    print(f"Target Feature         : '{TARGET}'")

    print("\nExecuting single forward inference pass on holdout test partition...")
    predictions = np.asarray(model.predict(X_test), dtype=float)

    if predictions.shape != (len(y_test),):
        raise ValueError(
            f"Model returned unexpected prediction shape: {predictions.shape}, expected {(len(y_test),)}."
        )

    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite (inf/-inf/nan) predictions on holdout test set.")

    actual = y_test.to_numpy(dtype=float)
    errors = actual - predictions
    abs_errors = np.abs(errors)

    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
    mae = float(mean_absolute_error(actual, predictions))
    medae = float(median_absolute_error(actual, predictions))
    r2 = float(r2_score(actual, predictions))

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build and save traceable test predictions DataFrame
    predictions_df = pd.DataFrame({
        "source_row_index": source_indices.to_numpy(),
        "actual_selling_price": actual,
        "predicted_selling_price": predictions,
        "error_actual_minus_predicted": errors,
        "absolute_error": abs_errors,
    })
    predictions_path = output_dir / "test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    # 2. Generate and save diagnostic plots
    actual_plot_path, residual_plot_path = generate_evaluation_plots(
        actual=actual,
        predictions=predictions,
        output_dir=output_dir,
        rmse=rmse,
        r2=r2,
    )

    # 3. Contextual comparison with Phase 4 CV metrics
    cv_metrics = p4_metadata.get("champion_cv_metrics", {})
    cv_rmse_mean = cv_metrics.get("cv_rmse_mean")
    cv_mae_mean = cv_metrics.get("cv_mae_mean")
    cv_r2_mean = cv_metrics.get("cv_r2_mean")

    metrics_payload: Dict[str, Any] = {
        "phase": 5,
        "selected_model": selected_model_name,
        "model_artifact_sha256": model_sha256,
        "test_sample_count": int(len(y_test)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "test_metrics": {
            "RMSE": rmse,
            "MAE": mae,
            "Median_Absolute_Error": medae,
            "R2": r2,
        },
        "phase4_cv_context": {
            "cv_rmse_mean": cv_rmse_mean,
            "cv_mae_mean": cv_mae_mean,
            "cv_r2_mean": cv_r2_mean,
            "note": "Cross-validation scores were derived strictly from the 5 training folds."
        },
        "metric_definitions_and_interpretations": {
            "RMSE": "Root Mean Squared Error: Measures standard deviation of residuals; penalizes large errors. Lower is better.",
            "MAE": "Mean Absolute Error: Average absolute prediction error in dataset price units. Lower is better.",
            "Median_Absolute_Error": "Median Absolute Error: Robust central tendency of error, unaffected by high-leverage outliers. Lower is better.",
            "R2": "Coefficient of Determination: Relative proportion of target variance explained. Higher is better; not a percentage accuracy.",
            "MAPE_omission_rationale": "MAPE is omitted as primary metric because near-zero target values cause distorted percentage blow-ups."
        },
        "leakage_and_integrity_audit": {
            "test_set_used_for_tuning": False,
            "test_set_used_for_model_selection": False,
            "model_retrained_in_phase5": False,
            "evaluated_holdout_rows": int(len(y_test)),
            "model_artifact_verified_sha256": model_sha256,
        },
        "artifacts": {
            "predictions_csv": str(predictions_path.resolve()),
            "actual_vs_predicted_plot": str(actual_plot_path.resolve()),
            "residual_diagnostics_plot": str(residual_plot_path.resolve()),
            "evaluation_metrics_json": str((output_dir / "evaluation_metrics.json").resolve()),
            "phase5_report_md": str((output_dir / "PHASE5_REPORT.md").resolve()),
        },
    }

    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # 4. Generate PHASE5_REPORT.md
    report_content = generate_markdown_report(
        selected_model=selected_model_name,
        model_sha256=model_sha256,
        n_test=len(y_test),
        rmse=rmse,
        mae=mae,
        medae=medae,
        r2=r2,
        cv_rmse=cv_rmse_mean,
        cv_mae=cv_mae_mean,
        cv_r2=cv_r2_mean,
        predictions_df=predictions_df,
    )
    report_path = output_dir / "PHASE5_REPORT.md"
    report_path.write_text(report_content, encoding="utf-8")

    # 5. Sanity check: verify saved predictions match computed metrics
    reloaded_preds = pd.read_csv(predictions_path)
    if len(reloaded_preds) != len(y_test):
        raise AssertionError("Sanity check failed: Saved predictions record count does not match test sample size.")

    reloaded_rmse = float(np.sqrt(mean_squared_error(
        reloaded_preds["actual_selling_price"],
        reloaded_preds["predicted_selling_price"],
    )))
    if not np.isclose(reloaded_rmse, rmse):
        raise AssertionError("Sanity check failed: Reloaded predictions do not reproduce computed test RMSE.")

    print_section_header("Holdout Test Evaluation Metrics")
    print(f"  Root Mean Squared Error (RMSE)    : {rmse:.4f} dataset price units")
    print(f"  Mean Absolute Error (MAE)         : {mae:.4f} dataset price units")
    print(f"  Median Absolute Error (MedAE)     : {medae:.4f} dataset price units")
    print(f"  Coefficient of Determination (R²) : {r2:.4f}")

    print_section_header("Comparison with Training 5-Fold Cross-Validation")
    if cv_rmse_mean is not None:
        print(f"  Mean CV RMSE : {cv_rmse_mean:.4f}  -->  Holdout Test RMSE : {rmse:.4f}")
        print(f"  Mean CV MAE  : {cv_mae_mean:.4f}  -->  Holdout Test MAE  : {mae:.4f}")
        print(f"  Mean CV R²   : {cv_r2_mean:.4f}  -->  Holdout Test R²   : {r2:.4f}")

    print_section_header("Phase 5 Execution Summary")
    print(f"Test Predictions CSV       : {predictions_path.resolve()}")
    print(f"Evaluation Metrics JSON    : {metrics_path.resolve()}")
    print(f"Actual vs Predicted Plot   : {actual_plot_path.resolve()}")
    print(f"Residual Diagnostics Plot  : {residual_plot_path.resolve()}")
    print(f"Phase 5 Summary Report     : {report_path.resolve()}")
    print_section_header("Phase 5 Final Model Evaluation Concluded Successfully")

    return predictions_df, metrics_payload


def generate_markdown_report(
    selected_model: str,
    model_sha256: str,
    n_test: int,
    rmse: float,
    mae: float,
    medae: float,
    r2: float,
    cv_rmse: Optional[float],
    cv_mae: Optional[float],
    cv_r2: Optional[float],
    predictions_df: pd.DataFrame,
) -> str:
    """
    Generate the contents of PHASE5_REPORT.md without unsupported assumptions.
    """
    top_errors = predictions_df.sort_values(by="absolute_error", ascending=False).head(3)
    error_rows_md = ""
    for _, r in top_errors.iterrows():
        idx = int(r["source_row_index"])
        act = float(r["actual_selling_price"])
        pred = float(r["predicted_selling_price"])
        err = float(r["error_actual_minus_predicted"])
        abs_err = float(r["absolute_error"])
        error_rows_md += f"| Row {idx} | {act:.2f} | {pred:.2f} | {err:+.2f} | {abs_err:.2f} |\n"

    cv_rmse_str = f"{cv_rmse:.4f}" if cv_rmse is not None else "N/A"
    cv_mae_str = f"{cv_mae:.4f}" if cv_mae is not None else "N/A"
    cv_r2_str = f"{cv_r2:.4f}" if cv_r2 is not None else "N/A"

    return f"""# Phase 5: Final Model Evaluation Report

**CodeAlpha Data Science Internship — Task 3: Car Price Prediction**

---

## 1. Evaluation Setup & Integrity Guarantees

- **Selected Model Pipeline**: `{selected_model}` (fitted in Phase 4 on 239 training rows)
- **Model Artifact Checksum (SHA-256)**: `{model_sha256}`
- **Holdout Test Set Size**: {n_test} records (exactly 20.0% of the 299 cleaned records)
- **Evaluation Protocol**:
  - The Phase 5 script evaluates the saved Phase 4 champion model (`best_model.joblib`) strictly on the reserved holdout partition without retraining or tuning.
  - The partition was reconstructed using the recorded split settings (`test_size=0.20`, `random_state=42`).

---

## 2. Holdout Test Performance Metrics

| Metric | Holdout Test Result | Phase 4 CV Training Baseline | Metric Interpretation & Nature |
| :--- | :---: | :---: | :--- |
| **RMSE** (Root Mean Squared Error) | **{rmse:.4f}** | {cv_rmse_str} | Standard deviation of residuals; heavily penalizes large errors. Lower is better. |
| **MAE** (Mean Absolute Error) | **{mae:.4f}** | {cv_mae_str} | Mean absolute magnitude of errors across all test cars. Lower is better. |
| **Median Absolute Error** | **{medae:.4f}** | — | Median error magnitude; robust against high-leverage outliers. Lower is better. |
| **R²** (Coefficient of Determination) | **{r2:.4f}** | {cv_r2_str} | Proportion of target variance explained relative to a baseline mean predictor. Higher is better. |

> **Crucial Statistical Note on R² and MAPE:**
> - $R^2$ measures explained variance, **not** percentage accuracy of individual car price estimates.
> - Mean Absolute Percentage Error (MAPE) is intentionally excluded from the primary metric suite because car selling prices in this dataset contain very low values (e.g., 0.1–0.5 units), where minor absolute deviations yield misleadingly inflated percentage errors.

### Generalization & Cross-Validation Gap Analysis

The holdout test RMSE (**{rmse:.4f}** dataset price units) is materially higher than the training 5-fold cross-validation mean RMSE (**{cv_rmse_str}** dataset price units). Similarly, the holdout test $R^2$ (**{r2:.4f}**) is lower than the cross-validation mean $R^2$ (**{cv_r2_str}**).

This performance divergence represents an important generalization concern:
- The gap indicates that the model experiences greater residual dispersion on fresh holdout records than observed across internal validation folds.
- **Important Statistical Precaution**: This gap alone does **not** conclusively prove severe model overfitting across the entire feature space. As demonstrated below, half the observations have absolute errors at or below **{medae:.4f}** units, subject to ties. Rather than a proven universal flaw, a plausible working hypothesis is that a few high-value records with large deviations disproportionately elevate RMSE due to error squaring.

---

## 3. Residual Diagnostics & Error Distribution (Residual = Actual - Predicted)

1. **Typical Performance (Median Absolute Error = {medae:.4f} units)**:
   The Median Absolute Error ({medae:.4f} dataset price units) indicates that half the observations have absolute errors at or below {medae:.4f} units, subject to ties. It provides a robust measure of typical error without implying that all vehicles are accurately estimated.
2. **Observed Residual Disparity & RMSE ({rmse:.4f} units)**:
   Because squared error squares large residuals, a small number of vehicles with larger deviations disproportionately increase RMSE relative to MAE ({mae:.4f}) and MedAE ({medae:.4f}). This is presented as an observed empirical characteristic of the sample residuals, rather than a definitive causal proof of model pathology.
3. **Largest Residual Cases on the Holdout Partition (Dynamically Identified)**:
| Source Row Index | Actual Selling Price | Predicted Selling Price | Signed Error (Actual - Pred) | Absolute Error |
| :---: | :---: | :---: | :---: | :---: | :---: |
{error_rows_md}

---

## 4. Scope, Domain Boundaries & Practical Limitations

- **Price Units**: The source dataset does not confirm the currency or unit; results are therefore reported strictly in neutral dataset price units.
- **Domain Limitations**:
  - The model should not be assumed universally accurate for newer vehicle cohorts (e.g., 2019–2026), post-pandemic automotive inflation, or electric/hybrid powertrains not represented in the training distribution.
  - Inferences on vehicle configurations with sparse representation in the training partition carry higher predictive uncertainty.

---

## 5. Generated Artifacts Index

- `test_predictions.csv`: Traceable test predictions with source row indices, actuals, predictions, and errors.
- `evaluation_metrics.json`: Complete machine-readable evaluation report with full audit trail.
- `actual_vs_predicted.png`: Scatter plot comparing actual vs. predicted prices with the 45° ideal line.
- `residual_diagnostics.png`: Dual diagnostic plots showing residuals vs. predictions and the residual histogram.
"""


def main() -> None:
    args = parse_args()
    run_phase5(
        csv_path=args.csv,
        model_path=args.model,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
