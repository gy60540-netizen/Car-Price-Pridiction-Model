"""
Phase 6: Car Price Prediction Inference Interface
Internship: CodeAlpha Data Science Internship - Task 3: Car Price Prediction

This module implements Phase 6:
  1. Loads the frozen Phase 4 fitted model pipeline without retraining or refitting.
  2. Defensively validates Phase 4 metadata against expected feature schema and target.
  3. Validates user input against strict statistical and dataset domain constraints:
       - Present_Price: finite float > 0
       - Driven_kms: finite float/int >= 0
       - Car_Age: finite int >= 0 (derived with historical reference year 2018)
       - Owner: finite int matching dataset values [0, 1, 3]
       - Fuel_Type: valid dataset category ('CNG', 'Diesel', 'Petrol')
       - Selling_type: valid dataset category ('Dealer', 'Individual')
       - Transmission: valid dataset category ('Automatic', 'Manual')
  4. Passes the validated 1-row DataFrame directly into the end-to-end Pipeline
     (encapsulating StandardScaler and OneHotEncoder).
  5. Outputs the predicted selling price in neutral dataset price units.
  6. Supports both interactive prompt mode and non-interactive command-line argument flags.
  7. Generates outputs/phase6/PHASE6_REPORT.md documenting interface usage, schema,
     reference-year limitations, and Phase 5 test performance context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
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
HISTORICAL_REFERENCE_YEAR: int = 2018


def print_section_header(title: str) -> None:
    """Print visual section header for terminal output."""
    separator = "=" * 72
    print(f"\n{separator}")
    print(f" {title.upper()}")
    print(separator)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 6 execution."""
    base = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Phase 6: Car Price Prediction Inference Interface using frozen Phase 4 pipeline."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=base / "outputs" / "phase4" / "best_model.joblib",
        help="Path to fitted Phase 4 model pipeline (best_model.joblib)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=base / "outputs" / "phase4" / "training_metadata.json",
        help="Path to Phase 4 training metadata (training_metadata.json)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=base / "outputs" / "phase3" / "cleaned_car_data.csv",
        help="Path to cleaned dataset used to validate categories (cleaned_car_data.csv)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=base / "outputs" / "phase6",
        help="Directory to store Phase 6 report (defaults to outputs/phase6)",
    )

    # Optional input parameters for non-interactive execution and testing
    parser.add_argument(
        "--present-price",
        type=float,
        default=None,
        help="Current showroom / present price of the vehicle (strictly positive float)",
    )
    parser.add_argument(
        "--driven-kms",
        type=float,
        default=None,
        help="Total kilometres driven (non-negative float)",
    )
    parser.add_argument(
        "--car-age",
        type=int,
        default=None,
        help=f"Age of vehicle relative to project reference year {HISTORICAL_REFERENCE_YEAR} ({HISTORICAL_REFERENCE_YEAR} - Year)",
    )
    parser.add_argument(
        "--owner",
        type=int,
        default=None,
        help="Number of previous owners (supported dataset values: 0, 1, 3)",
    )
    parser.add_argument(
        "--fuel-type",
        type=str,
        default=None,
        help="Fuel type ('CNG', 'Diesel', 'Petrol')",
    )
    parser.add_argument(
        "--selling-type",
        type=str,
        default=None,
        help="Selling seller type ('Dealer', 'Individual')",
    )
    parser.add_argument(
        "--transmission",
        type=str,
        default=None,
        help="Transmission type ('Automatic', 'Manual')",
    )

    return parser.parse_args()


def load_artifacts(
    model_path: Path,
    metadata_path: Path,
    csv_path: Path,
) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    """
    Load and defensively validate model pipeline, training metadata, and cleaned dataset.
    Raises actionable errors if files are missing or schema invariants are violated.
    """
    for path, label in [
        (model_path, "Fitted model pipeline"),
        (metadata_path, "Phase 4 training metadata"),
        (csv_path, "Cleaned dataset"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found at '{path.resolve()}'. Run previous phases first.")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if metadata.get("target") != TARGET:
        raise ValueError(
            f"Target mismatch: Expected '{TARGET}', but Phase 4 metadata specifies '{metadata.get('target')}'."
        )

    if metadata.get("features") != FEATURES:
        raise ValueError(
            f"Feature schema mismatch:\nExpected: {FEATURES}\nMetadata: {metadata.get('features')}"
        )

    df = pd.read_csv(csv_path)
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Cleaned dataset is missing required modeling columns: {missing_cols}")

    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        raise TypeError(f"Loaded object from '{model_path}' does not have a predict method.")

    selected_name = metadata.get("selected_model", "")
    estimator = getattr(model, "named_steps", {}).get("model") or getattr(model, "named_steps", {}).get("regressor")
    estimator_class_name = type(estimator).__name__ if estimator else ""
    if "Random Forest" in selected_name and "RandomForest" not in estimator_class_name:
        raise ValueError(
            f"Phase 4 metadata selected '{selected_name}', but loaded pipeline estimator is '{estimator_class_name}'."
        )
    if "RandomForest" in estimator_class_name and "Random Forest" not in selected_name:
        raise ValueError(
            f"Loaded pipeline estimator is '{estimator_class_name}', but Phase 4 metadata specifies '{selected_name}'."
        )

    return model, metadata, df


def get_supported_categories(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Extract sorted unique supported categories directly from the cleaned dataset."""
    return {
        col: sorted(df[col].dropna().astype(str).str.strip().unique().tolist())
        for col in CATEGORICAL_FEATURES
    }


def get_supported_owners(df: pd.DataFrame) -> List[int]:
    """Extract sorted supported integer owner counts directly from the cleaned dataset."""
    owner_values = pd.to_numeric(df["Owner"], errors="raise").dropna().unique()
    return sorted(int(v) for v in owner_values if float(v).is_integer() and v >= 0)


def validate_single_record(
    record: Dict[str, Any],
    supported_categories: Dict[str, List[str]],
    supported_owners: List[int],
) -> pd.DataFrame:
    """
    Validate all feature values for a single prediction record against domain constraints.
    Returns a 1-row pandas DataFrame strictly matching the feature schema.
    """
    # 1. Present_Price validation
    try:
        present_price = float(record["Present_Price"])
    except (ValueError, TypeError):
        raise ValueError(f"Present_Price must be a valid numeric value, got '{record.get('Present_Price')}'.")
    if not np.isfinite(present_price):
        raise ValueError("Present_Price must be finite. NaN, inf, and -inf are disallowed.")
    if present_price <= 0:
        raise ValueError(f"Present_Price must be strictly positive (> 0), got {present_price}.")

    # 2. Driven_kms validation
    try:
        driven_kms = float(record["Driven_kms"])
    except (ValueError, TypeError):
        raise ValueError(f"Driven_kms must be a valid numeric value, got '{record.get('Driven_kms')}'.")
    if not np.isfinite(driven_kms):
        raise ValueError("Driven_kms must be finite.")
    if driven_kms < 0:
        raise ValueError(f"Driven_kms cannot be negative, got {driven_kms}.")

    # 3. Car_Age validation
    try:
        car_age = float(record["Car_Age"])
    except (ValueError, TypeError):
        raise ValueError(f"Car_Age must be a valid integer, got '{record.get('Car_Age')}'.")
    if not np.isfinite(car_age) or not car_age.is_integer():
        raise ValueError(f"Car_Age must be a whole integer, got {car_age}.")
    car_age_int = int(car_age)
    if car_age_int < 0:
        raise ValueError(f"Car_Age cannot be negative, got {car_age_int}.")

    # 4. Owner validation
    try:
        owner = float(record["Owner"])
    except (ValueError, TypeError):
        raise ValueError(f"Owner must be an integer, got '{record.get('Owner')}'.")
    if not np.isfinite(owner) or not owner.is_integer():
        raise ValueError(f"Owner count must be a whole number, got {owner}.")
    owner_int = int(owner)
    if owner_int not in supported_owners:
        allowed = ", ".join(map(str, supported_owners))
        raise ValueError(
            f"Owner value {owner_int} is not supported by the dataset distribution. "
            f"Supported values: [{allowed}]."
        )

    # 5. Categorical features validation
    for col in CATEGORICAL_FEATURES:
        val = str(record.get(col, "")).strip()
        allowed = supported_categories.get(col, [])
        if val not in allowed:
            raise ValueError(
                f"Unsupported value '{val}' for '{col}'. Supported categories: {allowed}."
            )

    cleaned_dict = {
        "Present_Price": present_price,
        "Driven_kms": driven_kms,
        "Car_Age": car_age_int,
        "Owner": owner_int,
        "Fuel_Type": str(record["Fuel_Type"]).strip(),
        "Selling_type": str(record["Selling_type"]).strip(),
        "Transmission": str(record["Transmission"]).strip(),
    }

    return pd.DataFrame([cleaned_dict], columns=FEATURES)


def get_numeric_input(
    prompt: str,
    *,
    minimum: float,
    strictly_positive: bool = False,
    integer: bool = False,
) -> Union[float, int]:
    """Read and validate numeric value interactively."""
    while True:
        raw = input(prompt).strip()
        try:
            val = float(raw)
        except ValueError:
            print("  [ERROR] Invalid input. Please enter a valid numerical value.")
            continue

        if not np.isfinite(val):
            print("  [ERROR] Value must be finite (NaN and infinity are not allowed).")
            continue

        if strictly_positive and val <= minimum:
            print(f"  [ERROR] Value must be strictly greater than {minimum}.")
            continue
        elif not strictly_positive and val < minimum:
            print(f"  [ERROR] Value must be greater than or equal to {minimum}.")
            continue

        if integer and not val.is_integer():
            print("  [ERROR] Value must be a whole number (integer).")
            continue

        return int(val) if integer else val


def get_category_input(column: str, supported: List[str]) -> str:
    """Read and validate categorical choice interactively."""
    options_str = ", ".join(supported)
    print(f"\nSupported {column} options: [{options_str}]")
    while True:
        raw = input(f"Enter {column}: ").strip()
        matched = [opt for opt in supported if opt.lower() == raw.lower()]
        if matched:
            return matched[0]
        print(f"  [ERROR] Unsupported category '{raw}'. Please choose exactly one of: [{options_str}].")


def collect_interactive_input(
    supported_categories: Dict[str, List[str]],
    supported_owners: List[int],
) -> pd.DataFrame:
    """Interactively collect and validate vehicle attributes from stdin."""
    print_section_header("Interactive Vehicle Details Input")
    print(
        f"  IMPORTANT NOTE ON CAR AGE:\n"
        f"  Car_Age is defined relative to the project reference year {HISTORICAL_REFERENCE_YEAR}.\n"
        f"  Calculation: Car_Age = {HISTORICAL_REFERENCE_YEAR} - Manufacturing Year.\n"
        f"  (e.g., A 2014 manufacturing year corresponds to Car_Age = 4).\n"
    )

    present_price = get_numeric_input(
        "Enter Present_Price (in dataset price units, e.g. 5.59): ",
        minimum=0.0,
        strictly_positive=True,
    )

    driven_kms = get_numeric_input(
        "Enter Driven_kms (total kilometres driven, e.g. 27000): ",
        minimum=0.0,
    )

    car_age = get_numeric_input(
        f"Enter Car_Age (in years relative to reference year {HISTORICAL_REFERENCE_YEAR}, e.g. 4): ",
        minimum=0.0,
        integer=True,
    )

    owners_str = ", ".join(map(str, supported_owners))
    while True:
        owner = get_numeric_input(
            f"Enter Owner count (supported dataset values: [{owners_str}]): ",
            minimum=0.0,
            integer=True,
        )
        if owner in supported_owners:
            break
        print(f"  [ERROR] Owner value {owner} not in supported dataset categories: [{owners_str}].")

    cat_inputs: Dict[str, str] = {}
    for col in CATEGORICAL_FEATURES:
        cat_inputs[col] = get_category_input(col, supported_categories[col])

    raw_dict = {
        "Present_Price": present_price,
        "Driven_kms": driven_kms,
        "Car_Age": car_age,
        "Owner": owner,
        "Fuel_Type": cat_inputs["Fuel_Type"],
        "Selling_type": cat_inputs["Selling_type"],
        "Transmission": cat_inputs["Transmission"],
    }

    return validate_single_record(raw_dict, supported_categories, supported_owners)


def predict_price(model: Pipeline, user_row: pd.DataFrame) -> float:
    """
    Perform forward prediction using the loaded end-to-end Pipeline.
    Ensures input feature ordering is strictly preserved and output is finite.
    """
    preds = np.asarray(model.predict(user_row[FEATURES]), dtype=float)
    if preds.shape != (1,) or not np.isfinite(preds).all():
        raise ValueError(f"Pipeline returned unexpected or non-finite prediction: {preds}")
    return float(preds[0])


def load_phase5_metrics(report_dir: Path) -> Optional[Dict[str, float]]:
    """
    Attempt to load Phase 5 holdout evaluation metrics if available; returns None otherwise.
    Does not invent or assume values if the file is missing or corrupted.
    """
    phase5_file = report_dir.parent / "phase5" / "evaluation_metrics.json"
    if not phase5_file.is_file():
        return None
    try:
        with open(phase5_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("test_metrics")
    except (json.JSONDecodeError, OSError):
        return None


def generate_phase6_report(
    report_dir: Path,
    metadata: Dict[str, Any],
    phase5_metrics: Optional[Dict[str, float]] = None,
    sample_row: Optional[pd.DataFrame] = None,
    sample_prediction: Optional[float] = None,
    supported_categories: Optional[Dict[str, List[str]]] = None,
    supported_owners: Optional[List[int]] = None,
) -> Path:
    """
    Generate outputs/phase6/PHASE6_REPORT.md documenting usage, schema,
    reference year limitations, supported categories, and Phase 5 performance context.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "PHASE6_REPORT.md"

    selected_model = metadata.get("selected_model", "Random Forest")
    ref_year = metadata.get("reference_year", HISTORICAL_REFERENCE_YEAR)

    # Dynamic domain values from actual dataset distributions
    owner_domain_str = f"`{supported_owners}`" if supported_owners else "`[0, 1, 3]`"
    fuel_domain_str = f"`{supported_categories.get('Fuel_Type', [])}`" if supported_categories else "`['CNG', 'Diesel', 'Petrol']`"
    selling_domain_str = f"`{supported_categories.get('Selling_type', [])}`" if supported_categories else "`['Dealer', 'Individual']`"
    trans_domain_str = f"`{supported_categories.get('Transmission', [])}`" if supported_categories else "`['Automatic', 'Manual']`"

    # Dynamic metrics resolution without fabricating missing values
    if phase5_metrics:
        rmse_val = phase5_metrics.get("RMSE")
        mae_val = phase5_metrics.get("MAE")
        medae_val = phase5_metrics.get("Median_Absolute_Error", phase5_metrics.get("Median Absolute Error"))
        r2_val = phase5_metrics.get("R2")

        test_rmse_str = f"{rmse_val:.4f} dataset price units" if rmse_val is not None else "N/A"
        test_mae_str = f"{mae_val:.4f} dataset price units" if mae_val is not None else "N/A"
        test_medae_str = f"{medae_val:.4f} dataset price units" if medae_val is not None else "N/A"
        test_r2_str = f"{r2_val:.4f} (relative variance explained; not percentage accuracy)" if r2_val is not None else "N/A"

        medae_summary = (
            f"{medae_val:.4f} dataset price units"
            if medae_val is not None
            else "unavailable"
        )

        phase5_context_md = f"""- **Holdout Test Set Size**: 60 records (20% held-out test partition)
- **Holdout Test RMSE**: `{test_rmse_str}`
- **Holdout Test MAE**: `{test_mae_str}`
- **Holdout Test Median Absolute Error (MedAE)**: `{test_medae_str}`
- **Holdout Test R²**: `{test_r2_str}`

*Performance Observation*: For mainstream volume vehicles, the model predicts selling prices with a median absolute error of approximately {medae_summary}. High-present-price vehicles with large squared deviations elevate the overall RMSE."""
    else:
        phase5_context_md = "_Holdout evaluation metrics are currently unavailable or have not been executed yet._"

    sample_section = ""
    if sample_row is not None and sample_prediction is not None:
        row_dict = sample_row.iloc[0].to_dict()
        sample_section = f"""
---

## 6. Verified Prediction Execution

The following verified prediction was executed against the frozen Phase 4 pipeline:

| Input Feature | Value | Feature Description |
| :--- | :--- | :--- |
| `Present_Price` | `{row_dict.get('Present_Price')}` | Showroom price in dataset units |
| `Driven_kms` | `{row_dict.get('Driven_kms')}` | Total odometer distance |
| `Car_Age` | `{row_dict.get('Car_Age')}` | Relative to reference year {ref_year} |
| `Owner` | `{row_dict.get('Owner')}` | Previous owner category count |
| `Fuel_Type` | `{row_dict.get('Fuel_Type')}` | Primary powertrain fuel |
| `Selling_type` | `{row_dict.get('Selling_type')}` | Seller classification |
| `Transmission` | `{row_dict.get('Transmission')}` | Gearbox transmission |

**Predicted Selling Price**: **`{sample_prediction:.4f}`** dataset price units
"""

    report_content = f"""# Phase 6: Car Price Prediction Interface Report

**CodeAlpha Data Science Internship — Task 3: Car Price Prediction**

---

## 1. Interface & Model Artifact Overview

- **Model Pipeline Artifact**: `outputs/phase4/best_model.joblib`
- **Model Architecture**: `{selected_model}` (encapsulated inside a scikit-learn `Pipeline` with `StandardScaler` and `OneHotEncoder`)
- **Execution Mode**: **Inference Only**. The saved pipeline is evaluated as a frozen artifact; no retraining, parameter modification, or refitting occurs.
- **Price Units**: Predictions are output strictly in **dataset price units** (the source dataset does not confirm a currency/unit).

---

## 2. Input Feature Schema & Validation Invariants

Input records are defensively validated against the exact 7 modeling features in training order:

| Feature Name | Type | Validation Invariants & Constraints | Supported Domain Values |
| :--- | :--- | :--- | :--- |
| `Present_Price` | Numeric (float) | Strictly positive (> 0), finite | Continuously valued price units |
| `Driven_kms` | Numeric (float) | Non-negative (>= 0), finite | Odometer kilometres |
| `Car_Age` | Integer | Non-negative (>= 0), finite integer | Years relative to reference year {ref_year} |
| `Owner` | Integer | Supported training-dataset values (observed categories) | {owner_domain_str} |
| `Fuel_Type` | Categorical (str) | Exact match against dataset categories | {fuel_domain_str} |
| `Selling_type` | Categorical (str) | Exact match against dataset categories | {selling_domain_str} |
| `Transmission` | Categorical (str) | Exact match against dataset categories | {trans_domain_str} |

> **Domain Note on Features & Categories**:
> - Rejection of owner counts outside {owner_domain_str} reflects the domain observed in the training distribution, rather than implying other values are impossible in real-world scenarios.
> - The model's training distribution contains limited representation for high-owner counts and commercial vehicles.

---

## 3. Historical Reference Year Limitation

> **Important Domain Limitation on `Car_Age`:**
> In Phase 3 feature engineering, `Car_Age` was defined as:
> `Car_Age = 2018 - Year`
> Because 2018 is the maximum manufacturing year present in the dataset, `Car_Age` cannot be interpreted as age relative to the current calendar year. Inputting the age of a car relative to today's date would introduce severe temporal distribution shift and generate invalid estimates.

---

## 4. Phase 5 Holdout Performance Context

{phase5_context_md}

---

## 5. Usage Guide

### A. Non-Interactive CLI Mode (Automated / Scripted)
```powershell
python phase6_predict.py --present-price 5.59 --driven-kms 27000 --car-age 4 --owner 0 --fuel-type Petrol --selling-type Dealer --transmission Manual
```

### B. Interactive Mode
Run the script without parameter flags to launch the guided interactive prompt:
```powershell
python phase6_predict.py
```

### C. Incomplete Argument Handling
Supplying only a subset of feature flags triggers an actionable validation error listing all missing flags:
```powershell
python phase6_predict.py --present-price 5.59 --driven-kms 27000
# Output: Incomplete CLI input. Missing: --car-age, --owner, --fuel-type, --selling-type, --transmission
```
{sample_section}
---

## 7. Disclaimers & Practical Scope

1. **Not a Guaranteed Market Quote**: The predicted selling price reflects historical relationships learned from the 299-record dataset and is not an appraisal, commercial valuation, or guaranteed dealer offer.
2. **Out-of-Distribution Inputs**: Vehicles with features beyond the training range (e.g., modern electric cars, vintage pre-2000 models, or post-pandemic inflation) are outside the model's operational domain.
"""

    report_path.write_text(report_content, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()

    feature_flags = {
        "Present_Price": args.present_price,
        "Driven_kms": args.driven_kms,
        "Car_Age": args.car_age,
        "Owner": args.owner,
        "Fuel_Type": args.fuel_type,
        "Selling_type": args.selling_type,
        "Transmission": args.transmission,
    }

    supplied = {
        name: value
        for name, value in feature_flags.items()
        if value is not None
    }

    if supplied and len(supplied) != len(feature_flags):
        missing = [
            name for name, value in feature_flags.items()
            if value is None
        ]
        missing_flags = [
            "--" + name.lower().replace("_", "-")
            for name in missing
        ]
        raise SystemExit(
            "Incomplete CLI input. Missing: " + ", ".join(missing_flags)
        )

    model, metadata, df = load_artifacts(
        model_path=args.model,
        metadata_path=args.metadata,
        csv_path=args.csv,
    )

    supported_categories = get_supported_categories(df)
    supported_owners = get_supported_owners(df)

    print_section_header("CodeAlpha Task 3 - Phase 6: Car Price Prediction Interface")
    print(f"Model Pipeline   : {metadata.get('selected_model', 'Random Forest')} (Frozen Phase 4 Artifact)")
    print(f"Artifact Path    : {args.model.resolve()}")
    print("Execution Mode   : Forward Inference Only (No Retraining / Refitting)")
    print("Price Units      : Dataset Price Units (Not a Commercial Valuation)")

    if len(supplied) == len(feature_flags):
        user_row = validate_single_record(
            supplied, supported_categories, supported_owners
        )
    else:
        user_row = collect_interactive_input(
            supported_categories, supported_owners
        )

    predicted_price = predict_price(model, user_row)

    print("\n" + "-" * 72)
    print(" INPUT VEHICLE SPECIFICATION:")
    for col in FEATURES:
        print(f"   * {col:<15s}: {user_row.loc[0, col]}")
    print("-" * 72)
    print(f" PREDICTED SELLING PRICE : {predicted_price:.4f} dataset price units")
    print("-" * 72)
    print(
        f" [DOMAIN NOTICE]: Car_Age reflects historical reference year {HISTORICAL_REFERENCE_YEAR}.\n"
        f" This estimate is derived strictly from historical data and does not represent\n"
        f" a guaranteed real-world dealer quote or current market valuation."
    )
    print("=" * 72)

    # Generate the report only after successful prediction.
    report_path = generate_phase6_report(
        report_dir=args.report_dir,
        metadata=metadata,
        phase5_metrics=load_phase5_metrics(args.report_dir),
        sample_row=user_row,
        sample_prediction=predicted_price,
        supported_categories=supported_categories,
        supported_owners=supported_owners,
    )
    print(f"\nPhase 6 Report written to: {report_path.resolve()}\n")


if __name__ == "__main__":
    main()
