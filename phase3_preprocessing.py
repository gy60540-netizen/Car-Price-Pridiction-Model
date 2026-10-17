"""
Phase 3: Data Preprocessing & Feature Engineering
Internship: CodeAlpha Data Science Internship - Task 3: Car Price Prediction

This module implements Phase 3 only:
  1. Strict input validation, schema safety, and defensive data cleaning.
  2. Whitespace normalization on text/categorical features.
  3. Deduplication of exact duplicate listings for the modeling dataset (leaving raw CSV untouched).
  4. Domain boundary validation on numerical features (retaining legitimate luxury & mileage outliers).
  5. Feature Engineering: Parameterized, transparent vehicle depreciation metric ('Car_Age').
  6. Anti-Leakage Architecture: Blueprint ColumnTransformer/Pipeline definition without fitting on full dataset.
  7. Generation of modeling-ready CSV ('cleaned_car_data.csv') and machine-readable JSON summary.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------------------------------------------------------
# CONFIGURATION AND EXPECTED SCHEMA
# -------------------------------------------------------------------------

EXPECTED_COLUMNS: List[str] = [
    "Car_Name",
    "Year",
    "Selling_Price",
    "Present_Price",
    "Driven_kms",
    "Fuel_Type",
    "Selling_type",
    "Transmission",
    "Owner",
]

TARGET_COLUMN: str = "Selling_Price"

NUMERICAL_PREDICTORS: List[str] = [
    "Present_Price",
    "Driven_kms",
    "Car_Age",
    "Owner",
]

CATEGORICAL_PREDICTORS: List[str] = [
    "Fuel_Type",
    "Selling_type",
    "Transmission",
]

# High-cardinality features excluded from baseline regression feature matrix
EXCLUDED_PREDICTORS: List[str] = [
    "Car_Name",  # 98 distinct categories across 301 rows; causes extreme sparsity/overfitting
    "Year",      # Transformed into transparent 'Car_Age' feature to avoid collinearity
]

SANITY_MIN_YEAR: int = 1990
SANITY_MAX_YEAR: int = 2026


def print_section_header(title: str) -> None:
    """Print a clean visual section header for terminal output."""
    separator = "=" * 72
    print(f"\n{separator}")
    print(f" {title.upper()}")
    print(separator)


def resolve_default_dataset_path(base_dir: Optional[Path] = None) -> Path:
    """
    Reliably resolve dataset path, prioritizing 'car data(1).csv' then 'car data.csv'.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "car data(1).csv",
        base_dir / "car data.csv",
        base_dir / "car_data.csv",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return candidates[0]


def validate_input_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that input dataframe meets required column names, non-empty, and has no duplicates.
    """
    errors: List[str] = []
    if df is None or df.empty:
        errors.append("Dataset is empty or contains 0 records.")
        return False, errors

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required column(s): {missing_cols}")

    dup_cols = df.columns[df.columns.duplicated()].unique().tolist()
    if dup_cols:
        errors.append(f"Duplicate column headers found in dataset: {dup_cols}")

    return len(errors) == 0, errors


def clean_and_preprocess_dataset(
    df_raw: pd.DataFrame,
    reference_year: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute robust, defensive data cleaning and feature engineering for regression modeling:
      1. Normalizes string whitespace while preserving categories.
      2. Removes exact duplicate rows for modeling dataset (protects against train-test leakage).
      3. Validates finite numeric inputs and strictly positive target.
      4. Preserves domain outliers (luxury cars, delivery mileage) per Phase 2 EDA recommendations.
      5. Formulates transparent 'Car_Age' feature using configurable reference year.
      6. Segregates target, baseline predictors, and excluded high-cardinality features.
    """
    summary: Dict[str, Any] = {
        "decisions": [],
        "initial_row_count": len(df_raw),
        "initial_column_count": len(df_raw.columns),
    }

    # Step 1: Work on explicit copy to ensure raw data is never mutated
    df = df_raw.copy()

    # Step 2: Whitespace Normalization on Categorical/String Features
    string_cols = ["Car_Name", "Fuel_Type", "Selling_type", "Transmission"]
    ws_cleaned_details: Dict[str, int] = {}
    for col in string_cols:
        if col in df.columns:
            str_series = df[col].astype(str)
            stripped_series = str_series.str.strip()
            pad_count = int((str_series != stripped_series).sum())
            if pad_count > 0:
                ws_cleaned_details[col] = pad_count
            df[col] = stripped_series

    summary["whitespace_normalization"] = {
        "columns_checked": string_cols,
        "affected_counts": ws_cleaned_details,
        "policy": "Stripped leading/trailing whitespace without merging distinct model categories or altering semantics.",
    }
    summary["decisions"].append(
        f"Normalized whitespace in categorical columns (cleaned {sum(ws_cleaned_details.values())} padded values across {len(ws_cleaned_details)} column(s))."
    )

    # Step 3: Exact Duplicate Row Removal
    dup_mask = df.duplicated(keep="first")
    dup_indices = df[dup_mask].index.tolist()
    dup_count = len(dup_indices)
    dup_details: List[str] = []
    if dup_count > 0:
        for idx in dup_indices:
            row_data = df.loc[idx]
            match_indices = df[(df == row_data).all(axis=1)].index.tolist()
            original_idx = match_indices[0] if match_indices else "unknown"
            car_name = str(row_data.get("Car_Name", "Unknown"))
            year = str(row_data.get("Year", ""))
            dup_details.append(f"Row {idx} duplicates Row {original_idx} ({car_name} {year})")

    # Drop exact duplicates for modeling
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    summary["deduplication"] = {
        "duplicates_removed_count": dup_count,
        "duplicate_row_indices_removed": dup_indices,
        "duplicate_groups": dup_details,
        "policy": "Removed second instance of exact duplicate rows for modeling dataset to prevent train-test split leakage; original raw CSV remains unmodified.",
    }
    summary["decisions"].append(
        f"Removed {dup_count} exact duplicate row(s) (indices: {dup_indices}). Modeling dataset reduced from {summary['initial_row_count']} to {len(df)} records."
    )

    # Step 4: Strict Numeric Conversion and Infinite / NaN Validation
    num_cols = ["Year", "Selling_Price", "Present_Price", "Driven_kms", "Owner"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Check for NaNs or Infinities in critical modeling columns
    null_counts = {col: int(df[col].isna().sum()) for col in EXPECTED_COLUMNS if col in df.columns}
    inf_counts = {col: int(np.isinf(df[col]).sum()) for col in num_cols if col in df.columns}
    total_nulls = sum(null_counts.values())
    total_infs = sum(inf_counts.values())

    if total_nulls > 0:
        raise ValueError(
            f"[DATA QUALITY ERROR] Found {total_nulls} unconvertible NaN/missing value(s) in modeling dataset: {null_counts}. "
            "Policy: Reject silent imputation with zero."
        )
    if total_infs > 0:
        raise ValueError(
            f"[DATA QUALITY ERROR] Found {total_infs} infinite value(s) in numerical features: {inf_counts}. "
            "Policy: Reject infinite values for regression stability."
        )

    summary["missing_and_infinite_audit"] = {
        "null_counts": null_counts,
        "inf_counts": inf_counts,
        "status": "PASS (0 missing values, 0 infinite values)",
    }

    # Step 5: Domain Boundary & Target Sanity Validation
    # Target: Selling_Price
    sp = df[TARGET_COLUMN]
    if (sp <= 0).any():
        non_pos = df[sp <= 0]
        raise ValueError(
            f"[TARGET ERROR] Target '{TARGET_COLUMN}' contains non-positive values (n={len(non_pos)}). "
            "Car resale prices must be strictly positive finite values."
        )

    # Predictors: Present_Price, Driven_kms, Year, Owner
    pp = df["Present_Price"]
    if (pp <= 0).any():
        raise ValueError("[DOMAIN ERROR] 'Present_Price' contains non-positive values.")

    # Business Logic: Depreciation relationship check
    inverted_prices = df[df["Selling_Price"] > df["Present_Price"]]
    if not inverted_prices.empty:
        raise ValueError(
            f"[BUSINESS LOGIC ERROR] Found {len(inverted_prices)} record(s) where Selling_Price > Present_Price."
        )

    km = df["Driven_kms"]
    if (km < 0).any():
        raise ValueError("[DOMAIN ERROR] 'Driven_kms' contains negative values.")

    years = df["Year"]
    if (years < SANITY_MIN_YEAR).any() or (years > SANITY_MAX_YEAR).any():
        raise ValueError(f"[DOMAIN ERROR] 'Year' values fall outside sane boundary [{SANITY_MIN_YEAR}, {SANITY_MAX_YEAR}].")

    # Domain Outlier Retention Policy Documentation
    summary["outlier_handling_policy"] = {
        "Selling_Price_outliers": "17 statistical upper outliers (e.g. Land Cruiser, Fortuner) retained. These represent legitimate luxury vehicle depreciation dynamics.",
        "Present_Price_outliers": "14 statistical upper outliers retained as valid high-tier vehicle listings.",
        "Driven_kms_outliers": "8 upper outliers retained, including 500,000 km Activa delivery scooter. Not automatically dropped to preserve real-world mileage dispersion; robust/tree models handle without trimming.",
        "policy": "No automated outlier trimming applied. Retaining genuine market variation is essential for unbiased regression modeling.",
    }
    summary["decisions"].append(
        "Preserved all legitimate domain outliers (luxury price points and high-mileage records) without automated trimming."
    )

    # Step 6: Feature Engineering — Transparent 'Car_Age' Metric
    # Auto-resolve reference year if not explicitly provided
    dataset_max_year = int(df["Year"].max())
    if reference_year is None:
        ref_year = dataset_max_year
        ref_rationale = (
            f"Set to dataset max manufacturing year ({dataset_max_year}) to prevent lookahead calendar bias "
            "and guarantee non-negative vehicle age."
        )
    else:
        ref_year = reference_year
        ref_rationale = f"Explicitly configured caller reference year: {reference_year}."

    if ref_year < dataset_max_year:
        raise ValueError(
            f"[FEATURE ERROR] Reference year ({ref_year}) cannot be earlier than dataset max year ({dataset_max_year})."
        )

    df["Car_Age"] = ref_year - df["Year"].astype(int)
    if (df["Car_Age"] < 0).any():
        raise ValueError("[FEATURE ERROR] Engineered 'Car_Age' resulted in negative values.")

    summary["feature_engineering"] = {
        "new_feature": "Car_Age",
        "formula": f"Car_Age = {ref_year} - Year",
        "reference_year": ref_year,
        "reference_year_rationale": ref_rationale,
        "min_age": int(df["Car_Age"].min()),
        "max_age": int(df["Car_Age"].max()),
        "mean_age": round(float(df["Car_Age"].mean()), 2),
        "policy": "Derives intuitive vehicle age metric (newer cars = low age, older cars = high age) for monotonic depreciation modeling.",
    }
    summary["decisions"].append(
        f"Formulated 'Car_Age' feature using reference year {ref_year} (range: [{df['Car_Age'].min()}, {df['Car_Age'].max()}] years)."
    )

    # Step 7: Target & Feature Set Definition
    summary["feature_architecture"] = {
        "target_column": TARGET_COLUMN,
        "target_stats": {
            "min": round(float(sp.min()), 2),
            "max": round(float(sp.max()), 2),
            "mean": round(float(sp.mean()), 4),
            "median": round(float(sp.median()), 4),
            "std": round(float(sp.std()), 4),
        },
        "numerical_predictors": NUMERICAL_PREDICTORS,
        "categorical_predictors": CATEGORICAL_PREDICTORS,
        "excluded_from_baseline": EXCLUDED_PREDICTORS,
        "exclusion_justification": {
            "Car_Name": "High cardinality (98 distinct models in 299 rows). One-hot encoding creates excessive dimensionality (98 sparse columns), leading to severe overfitting. Showroom asset value ('Present_Price') serves as the continuous proxy for vehicle market segment.",
            "Year": "Fully captured with inverse sign by engineered 'Car_Age'. Including both would induce exact collinearity.",
        },
    }
    summary["decisions"].append(
        "Designated 'Selling_Price' strictly as target (excluded from predictors). Excluded 'Car_Name' from baseline predictor matrix to prevent high-cardinality overfitting."
    )

    summary["final_row_count"] = len(df)
    summary["final_column_count"] = len(df.columns)
    summary["final_columns"] = list(df.columns)

    return df, summary


def build_preprocessing_pipeline(
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Construct a clean, modular scikit-learn ColumnTransformer blueprint.

    CRITICAL LEAKAGE PREVENTION:
      This pipeline definition is instantiated WITHOUT being fitted on the full dataset.
      In Phase 4, it must be fitted strictly on X_train inside cross-validation splits.
        - Numerical features: Standardized with StandardScaler()
        - Categorical features: One-Hot Encoded with drop='first' (prevents dummy variable trap)
                                and handle_unknown='ignore'
    """
    if numerical_cols is None:
        numerical_cols = NUMERICAL_PREDICTORS
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_PREDICTORS

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def save_phase3_artifacts(
    df_cleaned: pd.DataFrame,
    summary: Dict[str, Any],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """
    Save the modeling-ready cleaned CSV and JSON preprocessing summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cleaned_car_data.csv"
    json_path = output_dir / "preprocessing_summary.json"

    df_cleaned.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return csv_path, json_path


def run_phase3(
    csv_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    reference_year: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Main orchestration routine for Phase 3 Data Preprocessing & Feature Engineering.
    """
    base_dir = Path(__file__).resolve().parent
    if csv_path is None:
        csv_path = resolve_default_dataset_path(base_dir)
    else:
        csv_path = Path(csv_path)

    if output_dir is None:
        output_dir = base_dir / "outputs" / "phase3"

    print_section_header("CodeAlpha Data Science Internship - Task 3: Car Price Prediction")
    print(" PHASE 3: DATA PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 72)
    print(f"Target CSV Path      : {csv_path.resolve()}")
    print(f"Output Directory     : {output_dir.resolve()}")

    if not csv_path.exists():
        raise FileNotFoundError(f"[FILE ERROR] Target CSV not found at: {csv_path.resolve()}")

    # Load raw CSV
    try:
        df_raw = pd.read_csv(csv_path)
    except Exception as err:
        print(f"[EXECUTION HALTED] Could not parse CSV dataset: {err}")
        return {}

    # Validate Schema
    valid, errors = validate_input_schema(df_raw)
    if not valid:
        print(f"[EXECUTION HALTED] Input schema validation failed:\n  - " + "\n  - ".join(errors))
        return {}

    # Execute Preprocessing & Feature Engineering
    try:
        df_cleaned, summary = clean_and_preprocess_dataset(
            df_raw, reference_year=reference_year
        )
    except (ValueError, KeyError) as err:
        print(f"[EXECUTION HALTED] Data quality or domain rule violation: {err}")
        return {}

    # Build Unfitted Preprocessing Pipeline Blueprint
    pipeline_blueprint = build_preprocessing_pipeline()
    summary["pipeline_blueprint"] = {
        "description": "scikit-learn ColumnTransformer (StandardScaler on numericals + OneHotEncoder(drop='first') on categoricals)",
        "fitted_on_full_dataset": False,
        "anti_leakage_note": "Pipeline is strictly unfitted. It will be fitted exclusively on training splits in Phase 4 modeling.",
    }

    # Save Artifacts
    saved_csv, saved_json = save_phase3_artifacts(df_cleaned, summary, output_dir)
    summary["saved_artifacts"] = {
        "cleaned_csv": str(saved_csv.resolve()),
        "summary_json": str(saved_json.resolve()),
    }

    # Print Comprehensive Terminal Validation Summary
    print_section_header("Phase 3 Execution & Validation Summary")
    print(f"Initial Observations     : {summary['initial_row_count']} rows × {summary['initial_column_count']} columns")
    print(f"Duplicates Removed       : {summary['deduplication']['duplicates_removed_count']} exact duplicate record(s)")
    print(f"Final Modeling Dataset   : {summary['final_row_count']} rows × {summary['final_column_count']} columns")
    print(f"Missing Values (NaN)     : {sum(summary['missing_and_infinite_audit']['null_counts'].values())}")
    print(f"Infinite Values (+/-Inf) : {sum(summary['missing_and_infinite_audit']['inf_counts'].values())}")
    print(f"Target Variable          : '{summary['feature_architecture']['target_column']}'")
    print(f"Target Validity Check    : Passed (Min: {summary['feature_architecture']['target_stats']['min']}, Max: {summary['feature_architecture']['target_stats']['max']}, Strictly Positive & Finite)")
    print(f"Engineered Feature       : 'Car_Age' (Reference Year: {summary['feature_engineering']['reference_year']}, Min: {summary['feature_engineering']['min_age']}, Max: {summary['feature_engineering']['max_age']})")
    print(f"Baseline Numerical Inputs: {summary['feature_architecture']['numerical_predictors']}")
    print(f"Baseline Categorical Inp : {summary['feature_architecture']['categorical_predictors']}")
    print(f"Excluded from Predictors : {summary['feature_architecture']['excluded_from_baseline']}")
    print(f"Anti-Leakage Enforcement : Pipeline instantiated but UNFITTED (prevents data leakage)")
    print("\n[SAVED ARTIFACTS]")
    print(f"  1. Cleaned Modeling CSV : {saved_csv.resolve()}")
    print(f"  2. JSON Preprocessing Doc: {saved_json.resolve()}")
    print("=" * 72)
    print(" PHASE 3 PREPROCESSING & FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("=" * 72 + "\n")

    return {
        "df_cleaned": df_cleaned,
        "summary": summary,
        "pipeline_blueprint": pipeline_blueprint,
    }


def main() -> None:
    """Command-line interface for standalone execution of Phase 3 Preprocessing."""
    parser = argparse.ArgumentParser(
        description="CodeAlpha Car Price Prediction - Phase 3 Data Preprocessing & Feature Engineering"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Custom path to CSV dataset (defaults to 'car data(1).csv' or 'car data.csv')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for Phase 3 artifacts (defaults to 'outputs/phase3')",
    )
    parser.add_argument(
        "--reference-year",
        type=int,
        default=None,
        help="Custom reference year for computing Car_Age (defaults to dataset max year)",
    )
    args = parser.parse_args()

    custom_path = Path(args.csv) if args.csv else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    results = run_phase3(
        csv_path=custom_path,
        output_dir=output_dir,
        reference_year=args.reference_year,
    )

    if not results:
        print("[ERROR] Phase 3 Data Preprocessing encountered an unrecoverable failure.")
        sys.exit(1)


if __name__ == "__main__":
    main()
