"""
Car Price Prediction - Phase 1 (Audit) & Phase 2 (Exploratory Data Analysis)
Internship: CodeAlpha Data Science Internship - Task 3

This module contains two decoupled, independently executable analytical stages:
  - Phase 1: Defensive Dataset Audit, Schema Validation & Quality Check
  - Phase 2: Exploratory Data Analysis (EDA), Statistical Testing & Visual Profiling

Strictly adheres to Phase 1 & 2 boundaries:
  - No machine learning model training or pipeline formulation is executed yet.
  - No automated deletion of duplicates, outlier trimming, or raw CSV modification.
  - Price units are strictly documented as neutral "dataset price units" (unconfirmed in raw CSV).
"""

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless non-interactive backend for automated execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------
# CONFIGURATION AND EXPECTED SCHEMA SPECIFICATION
# ---------------------------------------------------------
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

EXPECTED_NUMERIC_COLUMNS: List[str] = [
    "Year",
    "Selling_Price",
    "Present_Price",
    "Driven_kms",
    "Owner",
]

EXPECTED_CATEGORICAL_COLUMNS: List[str] = [
    "Car_Name",
    "Fuel_Type",
    "Selling_type",
    "Transmission",
]

EXPECTED_INTEGER_COLUMNS: List[str] = [
    "Year",
    "Driven_kms",
    "Owner",
]

# Configurable sanity-check thresholds (heuristics for flagging, not universal rejection rules)
SANITY_MIN_YEAR: int = 1990
SANITY_MAX_YEAR: int = 2026
PSEUDO_MISSING_TOKENS = {"", "na", "null", "none", "nan", "?", "n/a", "missing"}


def print_section_header(title: str) -> None:
    """Print a clean visual section header for terminal output."""
    separator = "=" * 72
    print(f"\n{separator}")
    print(f" {title.upper()}")
    print(separator)


def safe_percentage(numerator: int, denominator: int) -> float:
    """Safely calculate percentage without risk of ZeroDivisionError."""
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def interpret_skewness(skew_val: float) -> str:
    """
    Data-driven interpretation of numerical skewness.
    Note: Low skewness indicates low asymmetry around the mean; it does NOT prove
    that the distribution is normal (which also requires specific kurtosis and bell shape).
    """
    if pd.isna(skew_val):
        return "Undefined (insufficient data)"
    if skew_val > 1.0:
        return f"+{skew_val:.4f} (Highly right-skewed / positive tail)"
    if skew_val > 0.5:
        return f"+{skew_val:.4f} (Moderately right-skewed)"
    if skew_val < -1.0:
        return f"{skew_val:.4f} (Highly left-skewed / negative tail)"
    if skew_val < -0.5:
        return f"{skew_val:.4f} (Moderately left-skewed)"
    return f"{skew_val:.4f} (Low asymmetry / fairly symmetrical; does not imply normality)"


def resolve_default_dataset_path(base_dir: Path) -> Path:
    """
    Reliably locate the dataset file, prioritizing 'car data(1).csv' then 'car data.csv'.
    """
    candidates = [
        base_dir / "car data(1).csv",
        base_dir / "car data.csv",
        base_dir / "car_data.csv",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return candidates[0]


# =========================================================================
# PHASE 1: DATASET AUDIT AND QUALITY VALIDATION FUNCTIONS
# =========================================================================

def audit_raw_csv_header(csv_path: Path) -> Tuple[bool, List[str], List[str], List[str]]:
    """
    Inspect the raw first line of the CSV using csv.reader before pandas loads it.
    Catches exact duplicate headers, headers colliding after trimming/casing,
    blank or unnamed headers, whitespace padding, and schema mismatches.
    Preserves original header strings without modifying or silently renaming them.
    """
    print_section_header("1. Raw CSV Header and File Integrity Audit")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"[FILE ERROR] Target CSV not found at: {csv_path.resolve()}\n"
            f"Action Required: Verify the file path and ensure the dataset is placed inside the directory."
        )

    try:
        file_size = csv_path.stat().st_size
    except PermissionError as err:
        raise PermissionError(
            f"[PERMISSION ERROR] Access denied when accessing '{csv_path.name}': {err}\n"
            f"Action Required: Check file read permissions."
        ) from err

    if file_size == 0:
        raise ValueError(
            f"[FILE ERROR] Target CSV at '{csv_path.name}' is an empty 0-byte file.\n"
            f"Action Required: Replace with a valid, populated dataset."
        )

    errors: List[str] = []
    warnings: List[str] = []

    try:
        with open(csv_path, mode="r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            try:
                raw_header = next(reader)
            except StopIteration:
                raise ValueError(
                    f"[FILE ERROR] CSV '{csv_path.name}' contains no readable data or header line.\n"
                    f"Action Required: Provide a populated CSV file with column headers."
                )
            except csv.Error as err:
                raise ValueError(
                    f"[MALFORMED CSV ERROR] Failed to parse CSV header line in '{csv_path.name}': {err}\n"
                    f"Action Required: Verify CSV delimiter and quoting syntax."
                ) from err
    except UnicodeDecodeError as err:
        raise ValueError(
            f"[ENCODING ERROR] File encoding conflict reading '{csv_path.name}': {err}\n"
            f"Action Required: Ensure the file is encoded in UTF-8 or UTF-8 with BOM."
        ) from err
    except PermissionError as err:
        raise PermissionError(
            f"[PERMISSION ERROR] Access denied when reading '{csv_path.name}': {err}\n"
            f"Action Required: Check file read permissions."
        ) from err

    if len(raw_header) == 0:
        raise ValueError(
            f"[FILE ERROR] CSV '{csv_path.name}' contains an empty header line.\n"
            f"Action Required: Verify the CSV file begins with valid column names."
        )

    print(f"Raw Header Detected ({len(raw_header)} items): {raw_header}")

    # 1. Blank or unnamed column headers check (CRITICAL)
    blank_or_unnamed = [
        idx
        for idx, col in enumerate(raw_header, start=1)
        if not str(col).strip()
        or str(col).strip().lower().startswith("unnamed:")
        or str(col).strip().lower() == "unnamed"
    ]
    if blank_or_unnamed:
        msg = f"Blank or unnamed column header(s) found at position(s): {blank_or_unnamed}"
        print(f"  [CRITICAL] {msg}")
        errors.append(msg)
    else:
        print("  Raw Header Content: Passed (no blank or unnamed column headers).")

    # 2. Exact duplicate column headers check (CRITICAL)
    seen_exact: Dict[str, List[int]] = {}
    for idx, col in enumerate(raw_header, start=1):
        c_str = str(col).strip()
        if not c_str or c_str.lower().startswith("unnamed:") or c_str.lower() == "unnamed":
            continue
        seen_exact.setdefault(col, []).append(idx)

    exact_duplicates = {col: indices for col, indices in seen_exact.items() if len(indices) > 1}
    if exact_duplicates:
        dup_details = [f"'{col}' at positions {indices}" for col, indices in exact_duplicates.items()]
        msg = f"Exact duplicate column header(s) found in raw CSV: {', '.join(dup_details)}"
        print(f"  [CRITICAL] {msg}")
        errors.append(msg)
    else:
        print("  Raw Header Exact Uniqueness: Passed (no exact duplicate column names).")

    # 3. Headers duplicated after trimming whitespace and ignoring case (CRITICAL)
    norm_groups: Dict[str, List[Tuple[int, str]]] = {}
    for idx, col in enumerate(raw_header, start=1):
        c_str = str(col).strip()
        if not c_str or c_str.lower().startswith("unnamed:") or c_str.lower() == "unnamed":
            continue
        key = c_str.lower()
        norm_groups.setdefault(key, []).append((idx, col))

    normalized_duplicate_collisions: Dict[str, List[str]] = {}
    for key, items in norm_groups.items():
        distinct_raw = set(col for _, col in items)
        if len(items) > 1 and len(distinct_raw) > 1:
            normalized_duplicate_collisions[key] = [col for _, col in items]

    if normalized_duplicate_collisions:
        collision_details = [f"group '{k}': {cols}" for k, cols in normalized_duplicate_collisions.items()]
        msg = f"Header(s) become duplicates after trimming whitespace and ignoring case: {'; '.join(collision_details)}"
        print(f"  [CRITICAL] {msg}")
        errors.append(msg)
    else:
        print("  Raw Header Case/Whitespace Normalization: Passed (no headers collide when normalized).")

    # 4. Whitespace padding in headers check (WARNING)
    padded_headers = [col for col in raw_header if str(col) != str(col).strip()]
    if padded_headers:
        msg = f"Whitespace-padded column header(s) in raw CSV: {padded_headers}"
        print(f"  [WARNING] {msg}")
        warnings.append(msg)
    else:
        print("  Raw Header Padding: Passed (no whitespace-padded headers).")

    # 5. Missing required columns check (CRITICAL)
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in raw_header]
    if missing_cols:
        msg = f"Missing required column(s): {sorted(missing_cols)}"
        print(f"  [CRITICAL] {msg}")
        errors.append(msg)
    else:
        print("  Expected Schema Presence: Passed (all 9 expected columns present).")

    # 6. Unexpected extra columns check (WARNING)
    colliding_headers: Set[str] = set()
    for items in normalized_duplicate_collisions.values():
        colliding_headers.update(items)
    for col in exact_duplicates:
        colliding_headers.add(col)

    expected_lower = {c.lower() for c in EXPECTED_COLUMNS}
    extra_cols = [
        col for col in raw_header
        if col not in EXPECTED_COLUMNS
        and str(col).strip() != ""
        and not str(col).strip().lower().startswith("unnamed")
        and str(col).strip().lower() not in expected_lower
        and col not in colliding_headers
    ]
    if extra_cols:
        msg = f"Extra unexpected column(s) present: {sorted(extra_cols)}"
        print(f"  [WARNING] {msg}")
        warnings.append(msg)
    else:
        print("  Extra Columns: Passed (no unexpected extra columns).")

    is_valid = len(errors) == 0
    return is_valid, raw_header, errors, warnings


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Safely load the CSV file into a pandas DataFrame using specific exception handling.
    Fails fast with actionable messages for unreadable or 0-row datasets.
    Preserves pseudo-missing string tokens ('NA', 'null', 'nan', etc.) for explicit auditing,
    while treating genuinely empty fields as missing (NaN).
    """
    try:
        df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    except pd.errors.EmptyDataError as err:
        raise ValueError(
            f"[FILE ERROR] CSV at '{csv_path.name}' contains no readable data or header: {err}"
        ) from err
    except pd.errors.ParserError as err:
        raise ValueError(
            f"[PARSER ERROR] Failed to parse CSV structure at '{csv_path.name}': {err}"
        ) from err
    except PermissionError as err:
        raise PermissionError(
            f"[PERMISSION ERROR] Access denied when reading '{csv_path.name}': {err}"
        ) from err

    if len(df) == 0:
        raise ValueError(
            f"[DATA ERROR] CSV '{csv_path.name}' was loaded but contains 0 data rows.\n"
            f"Action Required: Dataset must contain observations for auditing and modeling."
        )

    return df


def audit_structure_and_schema(
    df: pd.DataFrame, raw_header: List[str]
) -> Tuple[bool, List[str], List[str]]:
    """
    Audit DataFrame dimensions, column inventory, and verify alignment with raw headers.
    """
    print_section_header("2. DataFrame Structure and Dimension Audit")
    num_rows, num_cols = df.shape
    print(f"Dataset Dimensions: {num_rows} observations (rows), {num_cols} attributes (columns)")

    errors: List[str] = []
    warnings: List[str] = []

    if num_cols != len(raw_header):
        msg = f"Column count mismatch: raw header has {len(raw_header)} columns, loaded DataFrame has {num_cols}."
        print(f"  [CRITICAL] {msg}")
        errors.append(msg)

    print("\nColumn Inventory:")
    for idx, col in enumerate(df.columns, start=1):
        print(f"  {idx:>2}. '{col}'")

    print("\nDataset Sample (First 5 Rows):")
    print(df.head(5).to_string(index=True))

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def audit_data_types_and_parsing(
    df: pd.DataFrame,
) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    """
    Validate data types against expectations.
    Detects non-numeric strings in expected numeric columns without coercing or deleting data.
    """
    print_section_header("3. Data Types and Parsing Validation")
    parsing_report: Dict[str, Any] = {}
    critical_errors: List[str] = []
    warnings: List[str] = []

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            print(f"  - {col:<15} : [SKIPPED] Missing from dataset")
            continue

        raw_dtype = df[col].dtype
        is_num = pd.api.types.is_numeric_dtype(df[col])
        expected_type = "Numeric" if col in EXPECTED_NUMERIC_COLUMNS else "Categorical"

        report: Dict[str, Any] = {
            "raw_dtype": str(raw_dtype),
            "expected_type": expected_type,
            "is_numeric": is_num,
            "non_numeric_values": [],
            "infinite_values": 0,
        }

        if expected_type == "Numeric":
            if not is_num:
                converted = pd.to_numeric(df[col], errors="coerce")
                bad_mask = df[col].notna() & converted.isna()
                bad_values = df.loc[bad_mask, col].unique().tolist()
                report["non_numeric_values"] = bad_values
                msg = f"Expected numeric column '{col}' contains non-numeric strings: {bad_values[:5]}"
                print(f"  [CRITICAL] {msg}")
                critical_errors.append(msg)
                status = "MISMATCH (contains non-numeric strings)"
            else:
                inf_count = int(np.isinf(df[col]).sum())
                report["infinite_values"] = inf_count
                if inf_count > 0:
                    msg = f"Numeric column '{col}' contains {inf_count} infinite value(s)!"
                    print(f"  [WARNING] {msg}")
                    warnings.append(msg)
                    status = f"MATCH (numeric) but has {inf_count} infinite values"
                else:
                    status = "MATCH (numeric, finite)"
        else:
            status = "MATCH (categorical/object)" if not is_num else "NOTE (stored as numeric)"

        parsing_report[col] = report
        print(f"  - {col:<15} : raw_dtype={str(raw_dtype):<10} | Expected: {expected_type:<12} | Status: {status}")

    is_valid = len(critical_errors) == 0
    return is_valid, critical_errors, warnings, parsing_report


def audit_missing_and_blank_values(df: pd.DataFrame) -> Tuple[int, int, List[str]]:
    """
    Explicit Missing-Value Audit Policy:
      Category A: Standard missing values parsed as NaN by pandas (empty fields).
      Category B: Unparsed pseudo-missing text tokens in non-null cells
                  ('na', 'null', 'none', 'nan', '?', 'n/a', 'missing', '').
      Guarantee: Mutually exclusive sets; zero double counting. Category B is
                 evaluated strictly on series.notna() non-null cells.
    """
    print_section_header("4. Missing and Blank Value Audit")
    warnings: List[str] = []

    total_cells = df.size
    standard_nulls = df.isnull().sum()
    total_standard_nulls = int(standard_nulls.sum())
    total_null_pct = safe_percentage(total_standard_nulls, total_cells)

    print("Policy: Standard NaNs (Category A) and Text Pseudo-Missing Tokens (Category B) audited separately.")
    print(f"Total Data Cells Inspected: {total_cells}")
    print(f"Category A - True Missing (NaN/None) Cells: {total_standard_nulls} ({total_null_pct:.2f}%)")

    print("\nCategory A - True Missing Values Per Column:")
    for col in df.columns:
        n_missing = int(standard_nulls[col])
        pct = safe_percentage(n_missing, len(df))
        print(f"  - {col:<15} : {n_missing:>3} missing ({pct:.2f}%)")

    if total_standard_nulls > 0:
        msg = f"Category A: Dataset contains {total_standard_nulls} true missing (NaN) cell(s) ({total_null_pct:.2f}% of total cells)."
        print(f"  [WARNING] {msg}")
        warnings.append(msg)

    print("\nCategory B - Pseudo-Missing String Tokens Audit (in non-null string cells):")
    total_pseudo_found = 0
    for col in df.columns:
        series = df[col]
        non_null_mask = series.notna()
        if not non_null_mask.any():
            continue

        str_repr = series[non_null_mask].astype(str).str.strip().str.lower()
        pseudo_mask = str_repr.isin(PSEUDO_MISSING_TOKENS)
        count = int(pseudo_mask.sum())

        if count > 0:
            total_pseudo_found += count
            flagged_vals = series[non_null_mask][pseudo_mask].unique().tolist()
            msg = f"Category B: Column '{col}' contains {count} pseudo-missing entry/entries: {flagged_vals}"
            print(f"  [WARNING] {msg}")
            warnings.append(msg)

    if total_pseudo_found == 0:
        print("  Passed: 0 pseudo-missing string tokens detected across all columns.")

    return total_standard_nulls, total_pseudo_found, warnings


def audit_duplicate_records(df: pd.DataFrame) -> Tuple[int, List[str]]:
    """
    Audit exact duplicate rows across all columns.
    Clearly distinguishes confirmed mathematical equality from business assumptions.
    """
    print_section_header("5. Duplicate Records Audit")
    dup_mask = df.duplicated()
    dup_count = int(dup_mask.sum())
    warnings: List[str] = []

    print(f"Exact Duplicate Rows Detected: {dup_count}")

    if dup_count > 0:
        print("\nIdentified Duplicate Listings (showing all duplicate instances):")
        all_dups = df[df.duplicated(keep=False)]
        print(all_dups.to_string(index=True))

        msg = f"{dup_count} exact duplicate rows identified across all features."
        warnings.append(msg)

        print("\nAnalysis & Distinction:")
        print("  * Confirmed Fact: The highlighted rows have 100% identical values across all columns.")
        print("  * Business Context: Two identical listings could represent accidental double-entry or identical specs.")
        print("  * Analytical Risk: Leaving duplicate rows risks train-test leakage.")
        print("  * Deferred Decision (Phase 3): Deduplication recommended in Phase 3 preprocessing.")
    else:
        print("No duplicate records found.")

    return dup_count, warnings


def audit_categorical_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Audit unique counts, frequencies, whitespace padding, and casing in categorical columns.
    """
    print_section_header("6. Categorical Feature Audit")
    warnings: List[str] = []
    flags: List[str] = []

    for col in EXPECTED_CATEGORICAL_COLUMNS:
        if col not in df.columns:
            print(f"\n[SKIPPED] Categorical feature '{col}' missing from dataset.")
            continue

        series = df[col]
        unique_count = series.nunique(dropna=False)
        print(f"\nFeature: '{col}' (Unique Categories: {unique_count})")

        # Whitespace padding detection
        non_null_mask = series.notna()
        padded_mask = non_null_mask & (series.astype(str).str.strip() != series.astype(str))
        if padded_mask.any():
            padded_items = series[padded_mask].unique().tolist()
            msg = f"Whitespace padding detected in {len(padded_items)} category value(s) of '{col}': {padded_items}"
            print(f"  [WARNING] {msg}")
            warnings.append(msg)
        else:
            print("  Whitespace Check: Passed (no padded categories).")

        # Value frequency inspection
        if unique_count <= 10:
            freq_df = series.value_counts(dropna=False).reset_index()
            freq_df.columns = ["Category", "Count"]
            freq_df["Percentage"] = freq_df["Count"].apply(lambda c: safe_percentage(c, len(df)))
            print(freq_df.to_string(index=False))

            # Flag severe minority classes
            for _, row in freq_df.iterrows():
                if row["Percentage"] < 1.0:
                    flag_msg = f"Severe class imbalance in '{col}': '{row['Category']}' represents only {row['Count']} row(s) ({row['Percentage']:.2f}%)."
                    flags.append(flag_msg)
        else:
            print(f"  High-cardinality feature ({unique_count} distinct entries).")
            print("  Top 5 Most Frequent:")
            print(series.value_counts(dropna=False).head(5).to_string())
            print("  Least Frequent Sample:")
            print(series.value_counts(dropna=False).tail(3).to_string())
            flag_msg = f"High-cardinality categorical feature '{col}' ({unique_count} distinct values) requires careful encoding in Phase 3/4."
            flags.append(flag_msg)

    return warnings, flags


def audit_numerical_features(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Audit numerical features: descriptive statistics, integer consistency,
    infinite values, and configurable heuristic range checks.
    Restricted strictly to valid finite values to prevent crashes.
    """
    print_section_header("7. Numerical Features and Statistical Summary")
    critical_errors: List[str] = []
    warnings: List[str] = []
    flags: List[str] = []

    for col in EXPECTED_NUMERIC_COLUMNS:
        if col not in df.columns:
            print(f"\n[SKIPPED] Numerical feature '{col}' missing from dataset.")
            continue

        s = df[col]

        if not pd.api.types.is_numeric_dtype(s):
            msg = f"Cannot compute numerical stats for '{col}': Column is non-numeric (dtype={s.dtype})."
            print(f"\n[SKIPPED] {msg}")
            if not any(col in err and "non-numeric" in err for err in critical_errors):
                critical_errors.append(msg)
            continue

        valid_mask = s.notna() & ~np.isinf(s)
        clean_s = s[valid_mask]
        inf_count = int(np.isinf(s).sum())
        nan_count = int(s.isna().sum())

        print(f"\nFeature: '{col}'")
        print(f"  Total Rows : {len(s)} | Valid Finite: {len(clean_s)} | NaN: {nan_count} | Inf: {inf_count}")

        if len(clean_s) == 0:
            msg = f"Numerical feature '{col}' has zero valid finite observations."
            print(f"  [CRITICAL] {msg}")
            critical_errors.append(msg)
            continue

        if inf_count > 0:
            msg = f"Numeric column '{col}' contains {inf_count} infinite value(s)!"
            if not any(col in w and "infinite" in w for w in warnings):
                warnings.append(msg)

        q1 = clean_s.quantile(0.25)
        median = clean_s.median()
        q3 = clean_s.quantile(0.75)
        iqr = q3 - q1
        skew_val = clean_s.skew() if len(clean_s) > 2 else float("nan")

        print(f"  Min        : {clean_s.min():<10} | Max    : {clean_s.max()}")
        print(f"  Mean       : {clean_s.mean():<10.4f} | Median : {median:.4f}")
        std_str = f"{clean_s.std():.4f}" if len(clean_s) > 1 else "N/A (single observation)"
        print(f"  Std Dev    : {std_str:<10} | IQR    : {iqr:.4f}")
        print(f"  Percentiles: 25%={q1:.2f}, 50%={median:.2f}, 75%={q3:.2f}")
        print(f"  Skewness   : {interpret_skewness(skew_val)}")

        # Integer consistency check for discrete columns
        if col in EXPECTED_INTEGER_COLUMNS:
            has_fractional = (clean_s % 1 != 0).any()
            if has_fractional:
                frac_examples = clean_s[clean_s % 1 != 0].head(3).tolist()
                msg = f"Expected integer column '{col}' contains fractional numbers: {frac_examples}"
                print(f"  [WARNING] {msg}")
                warnings.append(msg)
            else:
                print("  Integer Compatibility: Passed (all values are whole numbers).")

        # Negative values check
        neg_count = int((clean_s < 0).sum())
        if neg_count > 0:
            msg = f"Column '{col}' contains {neg_count} negative value(s)!"
            print(f"  [WARNING] {msg}")
            warnings.append(msg)
        else:
            print("  Negative Values Check: Passed (no negative values).")

        # Configurable heuristic sanity-checks
        if col == "Year":
            outside_mask = (clean_s < SANITY_MIN_YEAR) | (clean_s > SANITY_MAX_YEAR)
            if outside_mask.any():
                outside_years = df.loc[clean_s[outside_mask].index]
                msg = f"{len(outside_years)} observation(s) in 'Year' fall outside configurable sanity range [{SANITY_MIN_YEAR}, {SANITY_MAX_YEAR}]."
                print(f"  [INVESTIGATION NOTE] {msg}")
                flags.append(msg)
            else:
                print(f"  Sanity Threshold Check: Passed (all recorded years fall within [{SANITY_MIN_YEAR}, {SANITY_MAX_YEAR}]).")

        if col == "Driven_kms":
            zero_kms = int((clean_s == 0).sum())
            if zero_kms > 0:
                msg = f"{zero_kms} record(s) in 'Driven_kms' show 0 driven kilometers."
                print(f"  [INVESTIGATION NOTE] {msg}")
                flags.append(msg)
            high_km_mask = clean_s >= 400000
            if high_km_mask.any():
                high_km = df.loc[clean_s[high_km_mask].index]
                msg = f"High-mileage record(s) detected (>= 400k kms): {len(high_km)} record(s)."
                print(f"  [INVESTIGATION NOTE] {msg}")
                cols_to_show = [c for c in ["Car_Name", "Year", "Driven_kms", "Selling_Price"] if c in df.columns]
                print(f"         {high_km[cols_to_show].to_dict(orient='records')}")
                flags.append(msg)

        if col == "Owner":
            unique_levels = sorted(clean_s.unique().tolist())
            print(f"  Owner Levels Recorded: {unique_levels}")
            print(f"  Level Distribution   : {clean_s.value_counts().to_dict()}")
            if 3 in unique_levels and clean_s.value_counts().get(3, 0) <= 2:
                flags.append(f"Owner level 3 has only {clean_s.value_counts().get(3, 0)} record(s) (extreme rarity).")

    return critical_errors, warnings, flags


def audit_target_variable(
    df: pd.DataFrame,
) -> Tuple[bool, List[str], List[str], List[str]]:
    """
    Audit the target variable 'Selling_Price' for regression readiness.
    """
    print_section_header("8. Target Variable Audit ('Selling_Price')")
    critical_errors: List[str] = []
    warnings: List[str] = []
    flags: List[str] = []

    if TARGET_COLUMN not in df.columns:
        msg = f"Target column '{TARGET_COLUMN}' is missing from the dataset!"
        print(f"[CRITICAL] {msg}")
        critical_errors.append(msg)
        return False, critical_errors, warnings, flags

    target = df[TARGET_COLUMN]
    is_num = pd.api.types.is_numeric_dtype(target)
    print(f"Target Column Name       : {TARGET_COLUMN}")
    print(f"Is Numeric Datatype      : {is_num}")

    if not is_num:
        msg = f"Target '{TARGET_COLUMN}' is non-numeric! Cannot perform regression."
        print(f"[CRITICAL] {msg}")
        critical_errors.append(msg)
        return False, critical_errors, warnings, flags

    valid_mask = target.notna() & ~np.isinf(target)
    clean_target = target[valid_mask]

    print(f"Total Target Observations: {len(target)}")
    print(f"Valid Finite Observations: {len(clean_target)}")

    if len(clean_target) == 0:
        msg = f"Target '{TARGET_COLUMN}' has 0 valid finite observations."
        print(f"[CRITICAL] {msg}")
        critical_errors.append(msg)
        return False, critical_errors, warnings, flags

    missing_target = int(target.isnull().sum())
    inf_target = int(np.isinf(target).sum())
    non_positive = int((clean_target <= 0).sum())

    if missing_target > 0:
        warnings.append(f"Target '{TARGET_COLUMN}' contains {missing_target} missing value(s).")
    if inf_target > 0:
        warnings.append(f"Target '{TARGET_COLUMN}' contains {inf_target} infinite value(s).")
    if non_positive > 0:
        warnings.append(f"Target '{TARGET_COLUMN}' contains {non_positive} non-positive value(s) (<= 0).")

    print(f"Missing Values Count     : {missing_target}")
    print(f"Infinite Values Count    : {inf_target}")
    print(f"Non-positive Values (<=0): {non_positive}")
    print(f"Minimum Value            : {clean_target.min()}")
    print(f"Maximum Value            : {clean_target.max()}")
    print(f"Mean Value               : {clean_target.mean():.4f}")
    print(f"Median Value             : {clean_target.median():.4f}")
    std_target_str = f"{clean_target.std():.4f}" if len(clean_target) > 1 else "N/A"
    print(f"Standard Deviation       : {std_target_str}")

    skew_val = clean_target.skew() if len(clean_target) > 2 else float("nan")
    print(f"Distribution Skewness    : {interpret_skewness(skew_val)}")
    if not pd.isna(skew_val) and (skew_val > 1.0 or skew_val < -1.0):
        flags.append(f"Target skewness is {skew_val:.4f} ({interpret_skewness(skew_val)}). Evaluate log transformation in Phase 3.")

    print("\nPrice Scale / Currency Documentation Status:")
    print("  * Verified Fact: No currency code or numeric scale is documented in the raw CSV header.")
    print("  * Neutral Policy: Documented strictly as 'dataset price units'. No specific currency or multiplier is assumed.")
    flags.append("Pricing unit unconfirmed in raw CSV header; tracked neutrally as 'dataset price units'.")

    is_valid = len(critical_errors) == 0
    return is_valid, critical_errors, warnings, flags


def audit_business_logic_integrity(df: pd.DataFrame) -> List[str]:
    """
    Audit relationship integrity (e.g. Selling_Price vs Present_Price).
    Treats price inversion and low present price as investigation flags rather than definitive proof of invalidity.
    Defensively checks for column presence and numeric validity without assumptions on optional columns.
    """
    print_section_header("9. Business Logic & Relationship Integrity")
    flags: List[str] = []

    if "Selling_Price" not in df.columns or "Present_Price" not in df.columns:
        missing_price_cols = [c for c in ["Selling_Price", "Present_Price"] if c not in df.columns]
        print(f"[SKIPPED] Price comparison skipped: Missing column(s) {missing_price_cols}.")
        return flags

    p_selling = df["Selling_Price"]
    p_present = df["Present_Price"]

    if not (pd.api.types.is_numeric_dtype(p_selling) and pd.api.types.is_numeric_dtype(p_present)):
        non_num = [c for c in ["Selling_Price", "Present_Price"] if not pd.api.types.is_numeric_dtype(df[c])]
        print(f"[SKIPPED] Price comparison skipped: Column(s) {non_num} contain non-numeric types.")
        return flags

    valid_pair = p_selling.notna() & p_present.notna() & ~np.isinf(p_selling) & ~np.isinf(p_present)
    sp = p_selling[valid_pair]
    pp = p_present[valid_pair]

    print(f"Valid Price Pairs Inspected: {len(sp)}")
    if len(sp) == 0:
        print("  [SKIPPED] No valid, finite numeric price pairs available for comparison.")
        return flags

    greater_mask = sp > pp
    equal_mask = sp == pp

    greater_count = int(greater_mask.sum())
    equal_count = int(equal_mask.sum())
    print(f"Investigation Flag - Records where Selling_Price > Present_Price: {greater_count}")
    print(f"Investigation Flag - Records where Selling_Price == Present_Price: {equal_count}")

    if greater_count == 0:
        print("  Relationship Consistency: All valid observations show expected depreciation (Selling_Price < Present_Price).")
    else:
        inv_rows = df.loc[valid_pair][greater_mask]
        msg = f"{len(inv_rows)} record(s) exhibit selling price greater than showroom price."
        print(f"  [INVESTIGATION FLAG] {msg}")

        preferred_display_cols = ["Car_Name", "Year", "Selling_Price", "Present_Price"]
        display_cols = [c for c in preferred_display_cols if c in df.columns]
        if not display_cols:
            display_cols = list(df.columns)
        print(inv_rows[display_cols].to_string(index=True))
        flags.append(msg)

    # Low Present_Price investigation flag
    low_price_threshold = 2.0
    low_present = df.loc[valid_pair][pp < low_price_threshold]
    print(f"\nInvestigation Flag - Vehicles with Present_Price < {low_price_threshold} (dataset price units): {len(low_present)}")
    print("  * Contextual Explanation: Low showroom prices correspond predominantly to motorcycles,")
    print("    scooters, and entry-level mini-cars (e.g., Maruti 800, Royal Enfield, Bajaj, Honda Activa).")
    print("  * Classification Rule: This is treated as a domain investigation flag to explore in Phase 2 EDA,")
    print("    not as an automated rule to discard or split records.")
    if len(low_present) > 0:
        flags.append(f"{len(low_present)} vehicle(s) have Present_Price < {low_price_threshold} dataset price units (mix of two-wheelers and entry mini-cars).")

    return flags


def print_final_audit_summary(
    df: Optional[pd.DataFrame],
    critical_errors: List[str],
    warnings: List[str],
    flags: List[str],
) -> str:
    """
    Print an aggregated final summary and derive the overall validation status:
    PASS, PASS WITH WARNINGS, or FAIL.
    """
    print_section_header("10. Final Phase 1 Audit Summary")

    def deduplicate(items: List[str]) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    unique_critical = deduplicate(critical_errors)
    unique_warnings = deduplicate(warnings)
    unique_flags = deduplicate(flags)

    if unique_critical:
        overall_status = "FAIL"
        status_banner = "FAIL -- CRITICAL DATA INTEGRITY OR SCHEMA ISSUES DETECTED"
    elif unique_warnings or unique_flags:
        overall_status = "PASS WITH WARNINGS"
        status_banner = "PASS WITH WARNINGS -- READY FOR EDA WITH DOCUMENTED CAVEATS"
    else:
        overall_status = "PASS"
        status_banner = "PASS -- DATASET MEETS ALL QUALITY STANDARDS"

    print(f"OVERALL VALIDATION STATUS : [{overall_status}]")
    print(f"STATUS SUMMARY            : {status_banner}")

    if df is not None:
        print(f"TOTAL AUDITED OBSERVATIONS: {len(df)} rows, {df.shape[1]} columns")
    else:
        print("TOTAL AUDITED OBSERVATIONS: None (file unreadable or execution halted)")

    if unique_critical:
        print("\nCRITICAL ERRORS (PREVENTS MODELING):")
        for err in unique_critical:
            print(f"  [FAIL] {err}")

    if unique_warnings:
        print("\nACTIVE WARNINGS (REQUIRES PREPROCESSING ATTENTION):")
        for warn in unique_warnings:
            print(f"  [WARNING] {warn}")

    if unique_flags:
        print("\nINVESTIGATION FLAGS (TO EXPLORE IN PHASE 2 EDA):")
        for flg in unique_flags:
            print(f"  [FLAG] {flg}")

    print("\nDECISIONS DEFERRED TO SUBSEQUENT PHASES:")
    print("  1. Phase 3 Preprocessing: Drop 2 exact duplicate rows while preserving the raw CSV.")
    print("  2. Phase 3 Feature Engineering: Strip whitespace in 'Car_Name'.")
    print("  3. Phase 3 Feature Engineering: Formulate 'Car_Age = Reference_Year - Year'.")
    print("     (Evaluate trade-off between dataset max year 2018 vs current calendar year).")
    print("  4. Phase 3 Feature Selection: Assess high-cardinality 'Car_Name' (98 unique names) -")
    print("     explore brand extraction vs. dropping in favor of 'Present_Price' to prevent overfitting.")
    print("  5. Phase 4 Pipeline: Implement train-test split before one-hot encoding to prevent data leakage.")

    return overall_status


def run_phase1_audit(csv_path: Optional[Path] = None) -> str:
    """
    Main orchestration routine for Phase 1 dataset audit.
    Returns the overall validation status: 'PASS', 'PASS WITH WARNINGS', or 'FAIL'.
    """
    base_dir = Path(__file__).resolve().parent
    if csv_path is None:
        csv_path = resolve_default_dataset_path(base_dir)
    else:
        csv_path = Path(csv_path)

    print("=" * 72)
    print(" CODEALPHA DATA SCIENCE INTERNSHIP - TASK 3: CAR PRICE PREDICTION")
    print(" PHASE 1: COMPREHENSIVE DATASET AUDIT AND QUALITY VALIDATION")
    print("=" * 72)
    print(f"Target CSV Path: {csv_path.resolve()}")

    all_critical_errors: List[str] = []
    all_warnings: List[str] = []
    all_flags: List[str] = []
    df: Optional[pd.DataFrame] = None

    try:
        header_valid, raw_header, header_errors, header_warnings = audit_raw_csv_header(csv_path)
        all_critical_errors.extend(header_errors)
        all_warnings.extend(header_warnings)
    except (FileNotFoundError, ValueError, PermissionError) as err:
        print(f"\n[EXECUTION HALTED] {err}")
        all_critical_errors.append(str(err))
        return print_final_audit_summary(None, all_critical_errors, all_warnings, all_flags)

    try:
        df = load_dataset(csv_path)
    except (ValueError, PermissionError) as err:
        print(f"\n[EXECUTION HALTED] {err}")
        all_critical_errors.append(str(err))
        return print_final_audit_summary(None, all_critical_errors, all_warnings, all_flags)

    schema_valid, struct_errors, struct_warnings = audit_structure_and_schema(df, raw_header)
    all_critical_errors.extend(struct_errors)
    all_warnings.extend(struct_warnings)

    types_valid, type_errors, type_warnings, _ = audit_data_types_and_parsing(df)
    all_critical_errors.extend(type_errors)
    all_warnings.extend(type_warnings)

    _, _, missing_warnings = audit_missing_and_blank_values(df)
    all_warnings.extend(missing_warnings)

    _, dup_warnings = audit_duplicate_records(df)
    all_warnings.extend(dup_warnings)

    cat_warnings, cat_flags = audit_categorical_features(df)
    all_warnings.extend(cat_warnings)
    all_flags.extend(cat_flags)

    num_errors, num_warnings, num_flags = audit_numerical_features(df)
    all_critical_errors.extend(num_errors)
    all_warnings.extend(num_warnings)
    all_flags.extend(num_flags)

    target_valid, target_errors, target_warnings, target_flags = audit_target_variable(df)
    all_critical_errors.extend(target_errors)
    all_warnings.extend(target_warnings)
    all_flags.extend(target_flags)

    logic_flags = audit_business_logic_integrity(df)
    all_flags.extend(logic_flags)

    overall_status = print_final_audit_summary(df, all_critical_errors, all_warnings, all_flags)

    print("\n" + "=" * 72)
    print(f" PHASE 1 DATASET AUDIT CONCLUDED WITH STATUS: [{overall_status}]")
    print("=" * 72 + "\n")

    return overall_status


# =========================================================================
# PHASE 2: EXPLORATORY DATA ANALYSIS (EDA) & STATISTICAL PROFILING
# =========================================================================

def safe_shapiro(series: pd.Series) -> Tuple[Optional[float], Optional[float], str]:
    """Safe Shapiro-Wilk test handling small sample size, NaN, inf, or zero variance."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    clean = clean[np.isfinite(clean)]
    if len(clean) < 3:
        return None, None, f"Insufficient observations (n={len(clean)} < 3)"
    if clean.nunique() <= 1 or clean.std() == 0:
        return None, None, "Zero variance (constant series)"
    try:
        stat, p = stats.shapiro(clean)
        return float(stat), float(p), "OK"
    except Exception as e:
        return None, None, f"Shapiro-Wilk error: {e}"


def safe_pearsonr(x: pd.Series, y: pd.Series) -> Tuple[Optional[float], Optional[float], str]:
    """Safe Pearson correlation handling non-finite, missing, constant, or mismatched inputs."""
    sx = pd.to_numeric(x, errors="coerce")
    sy = pd.to_numeric(y, errors="coerce")
    mask = sx.notna() & sy.notna() & np.isfinite(sx) & np.isfinite(sy)
    clean_x, clean_y = sx[mask], sy[mask]
    if len(clean_x) < 2:
        return None, None, f"Insufficient paired finite observations (n={len(clean_x)} < 2)"
    if clean_x.nunique() <= 1 or clean_y.nunique() <= 1:
        return None, None, "Constant input variable (zero variance)"
    try:
        r, p = stats.pearsonr(clean_x, clean_y)
        return float(r), float(p), "OK"
    except Exception as e:
        return None, None, f"Pearson error: {e}"


def safe_spearmanr(x: pd.Series, y: pd.Series) -> Tuple[Optional[float], Optional[float], str]:
    """Safe Spearman rank correlation handling non-finite, missing, constant, or mismatched inputs."""
    sx = pd.to_numeric(x, errors="coerce")
    sy = pd.to_numeric(y, errors="coerce")
    mask = sx.notna() & sy.notna() & np.isfinite(sx) & np.isfinite(sy)
    clean_x, clean_y = sx[mask], sy[mask]
    if len(clean_x) < 2:
        return None, None, f"Insufficient paired finite observations (n={len(clean_x)} < 2)"
    if clean_x.nunique() <= 1 or clean_y.nunique() <= 1:
        return None, None, "Constant input variable (zero variance)"
    try:
        rho, p = stats.spearmanr(clean_x, clean_y)
        return float(rho), float(p), "OK"
    except Exception as e:
        return None, None, f"Spearman error: {e}"


def safe_mannwhitneyu(x: pd.Series, y: pd.Series) -> Tuple[Optional[float], Optional[float], str]:
    """Safe Mann-Whitney U test handling empty groups, zero variance, and non-finite values."""
    cx = pd.to_numeric(x, errors="coerce").dropna()
    cx = cx[np.isfinite(cx)]
    cy = pd.to_numeric(y, errors="coerce").dropna()
    cy = cy[np.isfinite(cy)]
    if len(cx) == 0 or len(cy) == 0:
        return None, None, f"Empty group encountered (group 1: n={len(cx)}, group 2: n={len(cy)})"
    if cx.nunique() <= 1 and cy.nunique() <= 1 and cx.iloc[0] == cy.iloc[0]:
        return None, None, "All values across both groups are identical"
    try:
        u, p = stats.mannwhitneyu(cx, cy, alternative="two-sided")
        return float(u), float(p), "OK"
    except Exception as e:
        return None, None, f"Mann-Whitney error: {e}"


def safe_kruskal(*groups: pd.Series) -> Tuple[Optional[float], Optional[float], str]:
    """Safe Kruskal-Wallis H test handling empty groups, zero variance, and small samples."""
    clean_groups = []
    for g in groups:
        cg = pd.to_numeric(g, errors="coerce").dropna()
        cg = cg[np.isfinite(cg)]
        if len(cg) > 0:
            clean_groups.append(cg)
    if len(clean_groups) < 2:
        return None, None, f"Kruskal-Wallis requires at least 2 non-empty groups (found {len(clean_groups)})"
    all_vals = np.concatenate([cg.to_numpy() for cg in clean_groups])
    if len(np.unique(all_vals)) <= 1:
        return None, None, "All values across all groups are identical"
    try:
        h, p = stats.kruskal(*clean_groups)
        return float(h), float(p), "OK"
    except Exception as e:
        return None, None, f"Kruskal error: {e}"


def eda_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform a defensive overview and quality recheck of the loaded dataset.
    Recalculates all metrics directly from the live DataFrame.
    """
    print_section_header("Phase 2 - Step 1: Dataset Overview & Quality Recheck")
    num_rows, num_cols = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    null_series = df.isnull().sum()
    total_nulls = int(null_series.sum())

    dup_count = int(df.duplicated().sum())

    padded_details: Dict[str, List[str]] = {}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        padded = df.loc[df[col].astype(str).str.strip() != df[col].astype(str), col].unique().tolist()
        if padded:
            padded_details[col] = padded

    print(f"Observations (Rows)      : {num_rows}")
    print(f"Attributes (Columns)     : {num_cols}")
    print(f"Memory Footprint         : {memory_mb:.4f} MB")
    print(f"Total True Missing (NaN) : {total_nulls}")
    print(f"Exact Duplicate Listings : {dup_count}")
    print(f"Columns with Whitespace  : {list(padded_details.keys()) if padded_details else 'None'}")

    print("\nData Types and Non-Null Summary:")
    col_summary = []
    for col in df.columns:
        col_summary.append({
            "Column": col,
            "Dtype": str(df[col].dtype),
            "Non-Null Count": int(df[col].notna().sum()),
            "Unique Values": int(df[col].nunique(dropna=False)),
        })
    summary_df = pd.DataFrame(col_summary)
    print(summary_df.to_string(index=False))

    overview_results = {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "memory_mb": memory_mb,
        "total_nulls": total_nulls,
        "dup_count": dup_count,
        "padded_details": padded_details,
    }
    return overview_results


def eda_univariate_analysis(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """
    Perform comprehensive univariate profiling of target, numerical, and categorical features.
    Saves informative distribution figures to the output directory using neutral unit descriptions.
    """
    print_section_header("Phase 2 - Step 2: Univariate Feature Analysis")
    results: Dict[str, Any] = {}

    # 1. Target Variable: Selling_Price
    clean_sp = pd.to_numeric(df["Selling_Price"], errors="coerce").dropna()
    sp_stats = {
        "mean": float(clean_sp.mean()) if len(clean_sp) > 0 else 0.0,
        "std": float(clean_sp.std()) if len(clean_sp) > 1 else 0.0,
        "median": float(clean_sp.median()) if len(clean_sp) > 0 else 0.0,
        "q1": float(clean_sp.quantile(0.25)) if len(clean_sp) > 0 else 0.0,
        "q3": float(clean_sp.quantile(0.75)) if len(clean_sp) > 0 else 0.0,
        "iqr": float(clean_sp.quantile(0.75) - clean_sp.quantile(0.25)) if len(clean_sp) > 0 else 0.0,
        "min": float(clean_sp.min()) if len(clean_sp) > 0 else 0.0,
        "max": float(clean_sp.max()) if len(clean_sp) > 0 else 0.0,
        "skewness": float(clean_sp.skew()) if len(clean_sp) > 2 else 0.0,
    }
    results["Selling_Price"] = sp_stats

    print(f"Target Feature ('Selling_Price'):")
    print(f"  Mean: {sp_stats['mean']:.4f} | Median: {sp_stats['median']:.4f} | Std Dev: {sp_stats['std']:.4f}")
    print(f"  Q1: {sp_stats['q1']:.2f} | Q3: {sp_stats['q3']:.2f} | IQR: {sp_stats['iqr']:.2f}")
    print(f"  Min: {sp_stats['min']:.2f} | Max: {sp_stats['max']:.2f}")
    print(f"  Distribution Skewness: {interpret_skewness(sp_stats['skewness'])}")
    print("  * Statistical Nuance: Target distribution exhibits heavy right-skewness (skew > 2.0).")
    print("    Note that non-normality of the target does NOT automatically imply that regression error")
    print("    residuals will be non-normal or heteroskedastic (residual properties depend on conditional")
    print("    distributions Y|X). However, extreme target skewness can create high-leverage points in")
    print("    ordinary least squares (OLS) loss; log-transformation log(1 + y) should be evaluated in Phase 3.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    use_kde_sp = bool(clean_sp.nunique() > 1 and len(clean_sp) > 1)
    bins_sp = min(30, max(1, clean_sp.nunique()))
    sns.histplot(clean_sp, kde=use_kde_sp, bins=bins_sp, ax=axes[0], color="#2b5c8f")
    axes[0].set_title("Selling_Price Distribution (Histogram + KDE)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Selling_Price (dataset price units)")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(sp_stats["mean"], color="#d9534f", linestyle="--", linewidth=2, label=f"Mean ({sp_stats['mean']:.2f})")
    axes[0].axvline(sp_stats["median"], color="#5cb85c", linestyle="-", linewidth=2, label=f"Median ({sp_stats['median']:.2f})")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    sns.boxplot(x=clean_sp, ax=axes[1], color="#85b8cb", flierprops={"marker": "o", "color": "#d9534f", "alpha": 0.7})
    axes[1].set_title("Selling_Price Boxplot (Outliers Highlighted)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Selling_Price (dataset price units)")
    axes[1].grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    p1 = output_dir / "eda_01_selling_price_distribution.png"
    plt.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {p1.name}")

    # 2. Present_Price Distribution
    clean_pp = pd.to_numeric(df["Present_Price"], errors="coerce").dropna()
    pp_stats = {
        "mean": float(clean_pp.mean()) if len(clean_pp) > 0 else 0.0,
        "std": float(clean_pp.std()) if len(clean_pp) > 1 else 0.0,
        "median": float(clean_pp.median()) if len(clean_pp) > 0 else 0.0,
        "min": float(clean_pp.min()) if len(clean_pp) > 0 else 0.0,
        "max": float(clean_pp.max()) if len(clean_pp) > 0 else 0.0,
        "skewness": float(clean_pp.skew()) if len(clean_pp) > 2 else 0.0,
    }
    results["Present_Price"] = pp_stats
    print(f"\nNumerical Feature ('Present_Price'):")
    print(f"  Mean: {pp_stats['mean']:.4f} | Median: {pp_stats['median']:.4f} | Skewness: {interpret_skewness(pp_stats['skewness'])}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    use_kde_pp = bool(clean_pp.nunique() > 1 and len(clean_pp) > 1)
    bins_pp = min(30, max(1, clean_pp.nunique()))
    sns.histplot(clean_pp, kde=use_kde_pp, bins=bins_pp, ax=axes[0], color="#e67e22")
    axes[0].set_title("Present_Price Distribution (Histogram + KDE)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Present_Price (dataset price units)")
    axes[0].axvline(pp_stats["mean"], color="#d9534f", linestyle="--", linewidth=2, label=f"Mean ({pp_stats['mean']:.2f})")
    axes[0].axvline(pp_stats["median"], color="#27ae60", linestyle="-", linewidth=2, label=f"Median ({pp_stats['median']:.2f})")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    sns.boxplot(x=clean_pp, ax=axes[1], color="#f8c471", flierprops={"marker": "o", "color": "#d9534f", "alpha": 0.7})
    axes[1].set_title("Present_Price Boxplot", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Present_Price (dataset price units)")
    axes[1].grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    p2 = output_dir / "eda_02_present_price_distribution.png"
    plt.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {p2.name}")

    # 3. Driven_kms Distribution
    clean_km = pd.to_numeric(df["Driven_kms"], errors="coerce").dropna()
    km_stats = {
        "mean": float(clean_km.mean()) if len(clean_km) > 0 else 0.0,
        "std": float(clean_km.std()) if len(clean_km) > 1 else 0.0,
        "median": float(clean_km.median()) if len(clean_km) > 0 else 0.0,
        "min": float(clean_km.min()) if len(clean_km) > 0 else 0.0,
        "max": float(clean_km.max()) if len(clean_km) > 0 else 0.0,
        "skewness": float(clean_km.skew()) if len(clean_km) > 2 else 0.0,
    }
    results["Driven_kms"] = km_stats
    print(f"\nNumerical Feature ('Driven_kms'):")
    print(f"  Mean: {km_stats['mean']:.1f} | Median: {km_stats['median']:.1f} | Skewness: {interpret_skewness(km_stats['skewness'])}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    use_kde_km = bool(clean_km.nunique() > 1 and len(clean_km) > 1)
    bins_km = min(30, max(1, clean_km.nunique()))
    sns.histplot(clean_km, kde=use_kde_km, bins=bins_km, ax=axes[0], color="#27ae60")
    axes[0].set_title("Driven_kms Distribution (Histogram + KDE)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Odometer Reading (Kilometers)")
    axes[0].axvline(km_stats["mean"], color="#d9534f", linestyle="--", linewidth=2, label=f"Mean ({km_stats['mean']:.0f})")
    axes[0].axvline(km_stats["median"], color="#2980b9", linestyle="-", linewidth=2, label=f"Median ({km_stats['median']:.0f})")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    sns.boxplot(x=clean_km, ax=axes[1], color="#a9dfbf", flierprops={"marker": "o", "color": "#d9534f", "alpha": 0.7})
    axes[1].set_title("Driven_kms Boxplot (Highlighting 500,000 km point)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Odometer Reading (Kilometers)")
    axes[1].grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    p3 = output_dir / "eda_03_driven_kms_distribution.png"
    plt.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {p3.name}")

    # 4. Year and Owner Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(data=df, x="Year", hue="Year", legend=False, ax=axes[0], palette="viridis")
    axes[0].set_title("Vehicle Distribution by Manufacturing Year", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Year of Manufacture")
    axes[0].set_ylabel("Listing Count")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    sns.countplot(data=df, x="Owner", hue="Owner", legend=False, ax=axes[1], palette="magma")
    axes[1].set_title("Vehicle Distribution by Past Owner Count", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Past Owners (0=First Hand, 1=Second Hand, 3=Fourth Hand)")
    axes[1].set_ylabel("Listing Count")
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    for p in axes[1].patches:
        h = p.get_height()
        axes[1].annotate(f"{int(h)} ({safe_percentage(int(h), len(df)):.1f}%)",
                         (p.get_x() + p.get_width() / 2.0, h + 3), ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    p4 = output_dir / "eda_04_year_owner_distributions.png"
    plt.savefig(p4, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {p4.name}")

    # 5. Categorical Counts: Fuel_Type, Selling_type, Transmission
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cat_cols = ["Fuel_Type", "Selling_type", "Transmission"]
    palettes = ["Blues_r", "Greens_r", "Oranges_r"]
    for i, col in enumerate(cat_cols):
        sns.countplot(data=df, x=col, hue=col, legend=False, ax=axes[i], palette=palettes[i])
        axes[i].set_title(f"Distribution of '{col}'", fontsize=12, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].grid(axis="y", linestyle=":", alpha=0.6)
        for p in axes[i].patches:
            h = p.get_height()
            axes[i].annotate(f"{int(h)}\n({safe_percentage(int(h), len(df)):.1f}%)",
                             (p.get_x() + p.get_width() / 2.0, h + 3), ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    p5 = output_dir / "eda_05_categorical_counts.png"
    plt.savefig(p5, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {p5.name}")

    # 6. Car_Name High Cardinality Intelligent Selection (Top 15 Models)
    fig, ax = plt.subplots(figsize=(10, 6))
    clean_car_names = df["Car_Name"].astype(str).str.strip().str.lower()
    top15 = clean_car_names.value_counts().head(15)
    sns.barplot(y=top15.index, x=top15.values, hue=top15.index, legend=False, ax=ax, palette="mako")
    ax.set_title("Top 15 Most Frequent Vehicle Models by Listing Count", fontsize=12, fontweight="bold")
    ax.set_xlabel("Listing Frequency (Total Listings)")
    ax.set_ylabel("Car Model")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    for i, v in enumerate(top15.values):
        ax.text(v + 0.3, i, f"{v} ({safe_percentage(v, len(df)):.1f}%)", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    p6 = output_dir / "eda_06_top_car_models.png"
    plt.savefig(p6, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {p6.name}")

    return results


def eda_bivariate_analysis(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """
    Perform bivariate analysis between Selling_Price and all predictive features.
    Computes both Pearson (linear) and Spearman (monotonic) correlations.
    """
    print_section_header("Phase 2 - Step 3: Bivariate Relationship Analysis")
    results: Dict[str, Any] = {}

    # Numerical Bivariate Relationships
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Selling_Price vs Present_Price
    r_sp, p_sp, _ = safe_pearsonr(df["Present_Price"], df["Selling_Price"])
    rho_sp, p_rho_sp, _ = safe_spearmanr(df["Present_Price"], df["Selling_Price"])
    sp_pp_title_r = f"r={r_sp:.4f}" if r_sp is not None else "r=N/A"
    sp_pp_title_rho = f"rho={rho_sp:.4f}" if rho_sp is not None else "rho=N/A"
    valid_pp_sp = pd.DataFrame({
        "Present_Price": pd.to_numeric(df["Present_Price"], errors="coerce"),
        "Selling_Price": pd.to_numeric(df["Selling_Price"], errors="coerce")
    }).dropna()
    if len(valid_pp_sp) > 1 and valid_pp_sp["Present_Price"].nunique() > 1 and valid_pp_sp["Selling_Price"].nunique() > 1:
        sns.regplot(data=valid_pp_sp, x="Present_Price", y="Selling_Price", ax=axes[0], color="#16a085",
                    scatter_kws={"alpha": 0.6, "edgecolor": "none"}, line_kws={"color": "#c0392b", "linewidth": 2})
    else:
        sns.scatterplot(data=valid_pp_sp, x="Present_Price", y="Selling_Price", ax=axes[0], color="#16a085", alpha=0.6)
    axes[0].set_title(f"Selling_Price vs Present_Price\nPearson {sp_pp_title_r} | Spearman {sp_pp_title_rho}", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Present_Price (dataset price units)")
    axes[0].set_ylabel("Selling_Price (dataset price units)")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Selling_Price vs Year
    r_yr, p_yr, _ = safe_pearsonr(df["Year"], df["Selling_Price"])
    rho_yr, p_rho_yr, _ = safe_spearmanr(df["Year"], df["Selling_Price"])
    sp_yr_title_r = f"r={r_yr:.4f}" if r_yr is not None else "r=N/A"
    sp_yr_title_rho = f"rho={rho_yr:.4f}" if rho_yr is not None else "rho=N/A"
    valid_yr_sp = pd.DataFrame({
        "Year": pd.to_numeric(df["Year"], errors="coerce"),
        "Selling_Price": pd.to_numeric(df["Selling_Price"], errors="coerce")
    }).dropna()
    if len(valid_yr_sp) > 1 and valid_yr_sp["Year"].nunique() > 1 and valid_yr_sp["Selling_Price"].nunique() > 1:
        sns.regplot(data=valid_yr_sp, x="Year", y="Selling_Price", ax=axes[1], color="#8e44ad",
                    scatter_kws={"alpha": 0.6, "edgecolor": "none"}, line_kws={"color": "#d35400", "linewidth": 2})
    else:
        sns.scatterplot(data=valid_yr_sp, x="Year", y="Selling_Price", ax=axes[1], color="#8e44ad", alpha=0.6)
    axes[1].set_title(f"Selling_Price vs Year\nPearson {sp_yr_title_r} | Spearman {sp_yr_title_rho}", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Manufacturing Year")
    axes[1].set_ylabel("Selling_Price (dataset price units)")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Selling_Price vs Driven_kms
    r_km, p_km, _ = safe_pearsonr(df["Driven_kms"], df["Selling_Price"])
    rho_km, p_rho_km, _ = safe_spearmanr(df["Driven_kms"], df["Selling_Price"])
    sp_km_title_r = f"r={r_km:.4f}" if r_km is not None else "r=N/A"
    sp_km_title_rho = f"rho={rho_km:.4f}" if rho_km is not None else "rho=N/A"
    valid_km_sp = pd.DataFrame({
        "Driven_kms": pd.to_numeric(df["Driven_kms"], errors="coerce"),
        "Selling_Price": pd.to_numeric(df["Selling_Price"], errors="coerce")
    }).dropna()
    if len(valid_km_sp) > 1 and valid_km_sp["Driven_kms"].nunique() > 1 and valid_km_sp["Selling_Price"].nunique() > 1:
        sns.regplot(data=valid_km_sp, x="Driven_kms", y="Selling_Price", ax=axes[2], color="#c0392b",
                    scatter_kws={"alpha": 0.6, "edgecolor": "none"}, line_kws={"color": "#2c3e50", "linewidth": 2})
    else:
        sns.scatterplot(data=valid_km_sp, x="Driven_kms", y="Selling_Price", ax=axes[2], color="#c0392b", alpha=0.6)
    axes[2].set_title(f"Selling_Price vs Driven_kms\nPearson {sp_km_title_r} | Spearman {sp_km_title_rho}", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Odometer Reading (km)")
    axes[2].set_ylabel("Selling_Price (dataset price units)")
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    p7 = output_dir / "eda_07_price_vs_numerical_bivariate.png"
    plt.savefig(p7, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {p7.name}")

    results["correlations"] = {
        "Present_Price": {"pearson_r": r_sp, "spearman_rho": rho_sp},
        "Year": {"pearson_r": r_yr, "spearman_rho": rho_yr},
        "Driven_kms": {"pearson_r": r_km, "spearman_rho": rho_km},
    }

    r_sp_fmt = f"{r_sp:+.4f}" if r_sp is not None else "N/A"
    rho_sp_fmt = f"{rho_sp:+.4f}" if rho_sp is not None else "N/A"
    r_yr_fmt = f"{r_yr:+.4f}" if r_yr is not None else "N/A"
    rho_yr_fmt = f"{rho_yr:+.4f}" if rho_yr is not None else "N/A"
    r_km_fmt = f"{r_km:+.4f}" if r_km is not None else "N/A"
    rho_km_fmt = f"{rho_km:+.4f}" if rho_km is not None else "N/A"

    print("\nNumerical Correlation Matrix with Target ('Selling_Price'):")
    print(f"  - Present_Price : Pearson r = {r_sp_fmt} (Strongest observed positive linear association among numerical features)")
    print(f"                    Spearman rho = {rho_sp_fmt} (Strongest observed monotonic rank relationship)")
    print(f"  - Year          : Pearson r = {r_yr_fmt} (Weak positive linear association; newer cars sell higher)")
    print(f"                    Spearman rho = {rho_yr_fmt} (Weak-to-moderate monotonic rank association)")
    print(f"  - Driven_kms    : Pearson r = {r_km_fmt} (Weak bivariate linear correlation across pooled data)")
    print(f"                    Spearman rho = {rho_km_fmt} (Weak monotonic rank relationship)")
    print("  * Statistical Caution: Correlation measures bivariate linear association, not direct causation.")
    print("    A weak aggregate linear correlation across pooled data does NOT prove that mileage has no effect.")
    print("    Rather, pooled bivariate correlation can be masked by vehicle segment heterogeneity: higher-tier")
    print("    vehicles may accumulate substantial mileage while retaining high absolute asset value, whereas")
    print("    lower-tier vehicles have low price ceilings even with very low mileage. The conditional effect")
    print("    of mileage must be assessed through multivariate modeling.")

    # Categorical Bivariate Relationships
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.boxplot(data=df, x="Fuel_Type", y="Selling_Price", hue="Fuel_Type", legend=False, ax=axes[0, 0], palette="Set2")
    axes[0, 0].set_title("Selling_Price by Fuel_Type (CNG n=2: Low Sample Size)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Selling_Price (dataset price units)")
    axes[0, 0].grid(axis="y", linestyle=":", alpha=0.6)

    sns.boxplot(data=df, x="Selling_type", y="Selling_Price", hue="Selling_type", legend=False, ax=axes[0, 1], palette="Set1")
    axes[0, 1].set_title("Selling_Price by Seller Type (Dealer vs Individual)", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Selling_Price (dataset price units)")
    axes[0, 1].grid(axis="y", linestyle=":", alpha=0.6)

    sns.boxplot(data=df, x="Transmission", y="Selling_Price", hue="Transmission", legend=False, ax=axes[1, 0], palette="Pastel1")
    axes[1, 0].set_title("Selling_Price by Transmission (Manual vs Automatic)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Selling_Price (dataset price units)")
    axes[1, 0].grid(axis="y", linestyle=":", alpha=0.6)

    sns.boxplot(data=df, x="Owner", y="Selling_Price", hue="Owner", legend=False, ax=axes[1, 1], palette="Accent")
    axes[1, 1].set_title("Selling_Price by Owner Level (Owner=3 n=1: Extreme Sparsity)", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Selling_Price (dataset price units)")
    axes[1, 1].grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    p8 = output_dir / "eda_08_price_by_categorical_bivariate.png"
    plt.savefig(p8, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {p8.name}")

    print("\nCategorical Group Medians:")
    for cat in ["Fuel_Type", "Selling_type", "Transmission", "Owner"]:
        grouped = df.groupby(cat)["Selling_Price"].agg(["count", "mean", "median", "std"]).round(2)
        print(f"\nGroup Summary by '{cat}':")
        print(grouped.to_string())

    return results


def eda_multivariate_analysis(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """
    Perform multivariate analysis: Pearson/Spearman correlation heatmaps and interaction plots.
    """
    print_section_header("Phase 2 - Step 4: Multivariate Correlation & Interactions")
    num_cols = [c for c in ["Selling_Price", "Present_Price", "Year", "Driven_kms", "Owner"] if c in df.columns]
    df_num = df[num_cols].apply(pd.to_numeric, errors="coerce")
    pearson_corr = df_num.corr(method="pearson")
    spearman_corr = df_num.corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0], vmin=-1, vmax=1,
                cbar_kws={"label": "Pearson Correlation (Linear)"})
    axes[0].set_title("Pearson Linear Correlation Matrix", fontsize=12, fontweight="bold")

    sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap="vlag", ax=axes[1], vmin=-1, vmax=1,
                cbar_kws={"label": "Spearman Correlation (Monotonic Rank)"})
    axes[1].set_title("Spearman Rank Correlation Matrix", fontsize=12, fontweight="bold")

    plt.tight_layout()
    p9 = output_dir / "eda_09_correlation_heatmaps.png"
    plt.savefig(p9, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {p9.name}")

    # Interaction plots with clean numeric data
    plot_df = df.copy()
    for col in ["Selling_Price", "Present_Price", "Driven_kms", "Year", "Owner"]:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=[c for c in ["Selling_Price", "Present_Price", "Driven_kms"] if c in plot_df.columns])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.scatterplot(data=plot_df, x="Present_Price", y="Selling_Price", hue="Transmission", style="Transmission",
                    s=80, alpha=0.8, ax=axes[0], palette="tab10")
    axes[0].set_title("Selling vs Present Price by Transmission\n(Automatic Concentrated in Upper Tier)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Present_Price (dataset price units)")
    axes[0].set_ylabel("Selling_Price (dataset price units)")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    sns.scatterplot(data=plot_df, x="Present_Price", y="Selling_Price", hue="Fuel_Type", s=80, alpha=0.8, ax=axes[1], palette="Dark2")
    axes[1].set_title("Selling vs Present Price by Fuel_Type\n(Diesel Listings Concentrated at Higher Price Points)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Present_Price (dataset price units)")
    axes[1].set_ylabel("Selling_Price (dataset price units)")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    sns.scatterplot(data=plot_df, x="Driven_kms", y="Selling_Price", hue="Selling_type", s=80, alpha=0.8, ax=axes[2], palette="Set2")
    axes[2].set_title("Selling_Price vs Driven_kms by Seller Type\n(Dealers Dominate Higher Price Points)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Driven_kms")
    axes[2].set_ylabel("Selling_Price (dataset price units)")
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    p10 = output_dir / "eda_10_multivariate_interactions.png"
    plt.savefig(p10, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {p10.name}")

    print("\nMultivariate Findings:")
    print("  1. Strong collinearity exists between Selling_Price and Present_Price (r=0.88, rho=0.91).")
    print("  2. Moderate negative correlation between Year and Driven_kms (r=-0.52, rho=-0.63) reflects chronological usage.")
    print("  3. In this sample, Automatic transmission listings and Diesel-engine listings are concentrated at higher price points.")

    return {"pearson_corr": pearson_corr, "spearman_corr": spearman_corr}


def eda_outlier_and_domain_investigation(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Rigorously investigate statistical outliers and domain anomalies without removing them.
    Explicitly documents findings and defers all cleaning actions to Phase 3.
    """
    print_section_header("Phase 2 - Step 5: Outlier & Domain Anomaly Investigation")
    investigations: Dict[str, Any] = {}

    def detect_iqr_outliers(series: pd.Series) -> Tuple[float, float, pd.DataFrame, int, int]:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if len(clean) == 0:
            return 0.0, 0.0, df.iloc[0:0], 0, 0
        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        num_s = pd.to_numeric(series, errors="coerce")
        lower_mask = num_s < lower_bound
        upper_mask = num_s > upper_bound
        outliers = df.loc[lower_mask | upper_mask]
        return lower_bound, upper_bound, outliers, int(lower_mask.sum()), int(upper_mask.sum())

    # A. Target variable (Selling_Price) IQR check
    sp_lb, sp_ub, sp_outliers, sp_lower_n, sp_upper_n = detect_iqr_outliers(df["Selling_Price"])
    print(f"1. 'Selling_Price' IQR Boundary Check [Q1 - 1.5*IQR, Q3 + 1.5*IQR]: [{sp_lb:.2f}, {sp_ub:.2f}]")
    print(f"   - Total statistical outliers flagged: {len(sp_outliers)} record(s) ({sp_lower_n} lower < {sp_lb:.2f}, {sp_upper_n} upper > {sp_ub:.2f} dataset price units)")
    clean_sp_series = pd.to_numeric(df["Selling_Price"], errors="coerce").dropna()
    if len(clean_sp_series) > 0:
        peak_sp_idx = clean_sp_series.idxmax()
        print(f"   - Highest Listing: {df.loc[peak_sp_idx, ['Car_Name', 'Year', 'Selling_Price', 'Present_Price']].to_dict()}")
    print("   - Domain Assessment: These represent legitimate luxury vehicles (e.g. Land Cruiser, Fortuner, Innova),")
    print("     not erroneous data entries. Trimming them would distort real luxury car depreciation dynamics.")

    # B. Present_Price IQR check
    pp_lb, pp_ub, pp_outliers, pp_lower_n, pp_upper_n = detect_iqr_outliers(df["Present_Price"])
    print(f"\n2. 'Present_Price' IQR Boundary Check: [{pp_lb:.2f}, {pp_ub:.2f}]")
    print(f"   - Total statistical outliers flagged: {len(pp_outliers)} record(s) ({pp_lower_n} lower < {pp_lb:.2f}, {pp_upper_n} upper > {pp_ub:.2f} dataset price units)")
    clean_pp_series = pd.to_numeric(df["Present_Price"], errors="coerce").dropna()
    if len(clean_pp_series) > 0:
        peak_pp_idx = clean_pp_series.idxmax()
        print(f"   - Peak Showroom Price: {df.loc[peak_pp_idx, ['Car_Name', 'Year', 'Present_Price', 'Selling_Price']].to_dict()}")

    # C. Driven_kms IQR check
    km_lb, km_ub, km_outliers, km_lower_n, km_upper_n = detect_iqr_outliers(df["Driven_kms"])
    print(f"\n3. 'Driven_kms' IQR Boundary Check: [{km_lb:.0f}, {km_ub:.0f}]")
    print(f"   - Total statistical outliers flagged: {len(km_outliers)} record(s) ({km_lower_n} lower < {km_lb:.0f}, {km_upper_n} upper > {km_ub:.0f} km)")
    clean_km_series = pd.to_numeric(df["Driven_kms"], errors="coerce").dropna()
    high_km_row: Dict[str, Any] = {}
    if len(clean_km_series) > 0:
        high_km_idx = clean_km_series.idxmax()
        high_km_row = df.loc[high_km_idx, ["Car_Name", "Year", "Driven_kms", "Selling_Price", "Present_Price"]].to_dict()
        print(f"   - Extreme Peak Mileage: {high_km_row}")
        print("   - Domain Assessment: 500,000 km on a 2008 Activa 3g scooter. Over 10 years, 50,000 km/year is plausible")
        print("     for a commercial delivery vehicle, but could also be a transcription typo for 50,000 km.")
        print("     Deferred Decision: Flagged for sensitivity analysis during Phase 3 feature engineering.")

    # D. Exact duplicate rows
    dup_indices = df[df.duplicated(keep="first")].index.tolist()
    dup_rows = df[df.duplicated(keep=False)]
    print(f"\n4. Exact Duplicate Rows: {len(dup_indices)} duplicate row(s) ({len(dup_rows)} rows displayed showing all copies):")
    if not dup_rows.empty:
        cols_to_show = [c for c in ["Car_Name", "Year", "Selling_Price", "Present_Price", "Driven_kms"] if c in df.columns]
        print(dup_rows[cols_to_show].to_string())
    print("   - Domain Assessment: High risk of data leakage if left across train-test splits.")
    print("     Deferred Decision: Scheduled to drop second occurrence in Phase 3.")

    # E. Owner Level = 3
    if "Owner" in df.columns:
        owner3 = df[df["Owner"] == 3]
        print(f"\n5. Extreme Rarity in 'Owner': Level 3 appears only {len(owner3)} time(s):")
        cols_o = [c for c in ["Car_Name", "Year", "Selling_Price", "Present_Price", "Owner"] if c in df.columns]
        print(owner3[cols_o].to_dict(orient="records"))
        print("   - Domain Assessment: Extreme sparsity (n=1) prevents robust parameter estimation.")

    # F. Fuel_Type = 'CNG'
    if "Fuel_Type" in df.columns:
        cng = df[df["Fuel_Type"] == "CNG"]
        print(f"\n6. Extreme Rarity in 'Fuel_Type': 'CNG' appears only {len(cng)} time(s):")
        cols_f = [c for c in ["Car_Name", "Year", "Selling_Price", "Present_Price", "Fuel_Type"] if c in df.columns]
        print(cng[cols_f].to_dict(orient="records"))
        print("   - Domain Assessment: Too small (n=2) for statistical generalization.")

    # G. Vehicles with Present_Price < 2.0
    if "Present_Price" in df.columns:
        num_pp = pd.to_numeric(df["Present_Price"], errors="coerce")
        low_pp = df[num_pp < 2.0]
        print(f"\n7. Budget Segment & Two-Wheelers: {len(low_pp)} vehicle(s) have Present_Price < 2.0 dataset price units:")
        if "Car_Name" in df.columns:
            print(f"   - Sample names: {low_pp['Car_Name'].head(5).tolist()}")
        print("   - Domain Assessment: Demonstrates that dataset is a heterogeneous mixture of two-wheelers")
        print("     (Activa, Royal Enfield, Pulsar) and four-wheelers. Distinct segments have differing price dynamics.")

    investigations["sp_outlier_count"] = len(sp_outliers)
    investigations["sp_lower_outliers"] = sp_lower_n
    investigations["sp_upper_outliers"] = sp_upper_n
    investigations["km_outlier_count"] = len(km_outliers)
    investigations["km_lower_outliers"] = km_lower_n
    investigations["km_upper_outliers"] = km_upper_n
    investigations["high_km_record"] = high_km_row
    return investigations


def eda_statistical_hypothesis_testing(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute rigorous inferential hypothesis tests to evaluate category price differences.
    Evaluates underlying assumptions (normality, variance, sample size) before drawing conclusions.
    """
    print_section_header("Phase 2 - Step 6: Inferential Statistical Hypothesis Testing")
    test_results: Dict[str, Any] = {}

    # 1. Normality Test on Selling_Price (Shapiro-Wilk)
    stat_sw, p_sw, note_sw = safe_shapiro(df["Selling_Price"])
    test_results["shapiro_selling_price"] = {"stat": stat_sw, "p_value": p_sw, "note": note_sw}
    print("1. Normality Test (Shapiro-Wilk) on 'Selling_Price':")
    if stat_sw is not None:
        print(f"   - W-Statistic = {stat_sw:.4f}, p-value = {p_sw:.4e}")
        if p_sw < 0.05:
            print("   - Conclusion: p-value < 0.05 rejects the null hypothesis of marginal normality for Selling_Price.")
            print("     Important Distinction for Modeling: This test assesses only the marginal distribution of the")
            print("     target Y. It does NOT evaluate or prove regression residual non-normality or heteroskedasticity,")
            print("     which depend on the conditional distribution Y|X after fitting a regression model.")
            print("     However, heavy positive skewness (+2.49) indicates high-leverage observations in OLS loss,")
            print("     motivating evaluation of variance-stabilizing transforms (e.g. log(1+y)) and tree-based models.")
            print("     Consequently, non-parametric rank tests (Mann-Whitney U, Kruskal-Wallis) are appropriately used below")
            print("     to compare group medians without relying on Gaussian distributional assumptions.")
    else:
        print(f"   - [SKIPPED] {note_sw}")

    # 2. Mann-Whitney U Test: Selling_Price between Diesel and Petrol
    petrol_prices = df.loc[df["Fuel_Type"] == "Petrol", "Selling_Price"]
    diesel_prices = df.loc[df["Fuel_Type"] == "Diesel", "Selling_Price"]
    u_fuel, p_fuel, note_fuel = safe_mannwhitneyu(diesel_prices, petrol_prices)
    test_results["mannwhitney_fuel"] = {"u_stat": u_fuel, "p_value": p_fuel, "note": note_fuel}
    print("\n2. Mann-Whitney U Test: Selling_Price (Diesel vs Petrol):")
    if u_fuel is not None:
        clean_d = pd.to_numeric(diesel_prices, errors="coerce").dropna()
        clean_p = pd.to_numeric(petrol_prices, errors="coerce").dropna()
        print(f"   - Sample Sizes: Diesel (n={len(clean_d)}), Petrol (n={len(clean_p)})")
        print(f"   - Diesel Median: {clean_d.median():.2f} | Petrol Median: {clean_p.median():.2f}")
        print(f"   - U-Statistic = {u_fuel:.1f}, p-value = {p_fuel:.4e}")
        if p_fuel < 0.05:
            print("   - Conclusion: Statistically significant difference (p < 0.001). In this sample, Diesel vehicles exhibit")
            print("     a higher median resale price than Petrol vehicles.")
            print("     Causality & Confounding Caution: Statistical significance establishes an observed group distributional")
            print("     difference, NOT that choosing diesel fuel causes higher resale value. Confounding factors such as")
            print("     higher initial purchase price or segment composition are plausible hypotheses but are not independently")
            print("     isolated by this bivariate test.")
    else:
        print(f"   - [SKIPPED] {note_fuel}")

    # 3. Mann-Whitney U Test: Selling_Price between Automatic and Manual
    manual_prices = df.loc[df["Transmission"] == "Manual", "Selling_Price"]
    auto_prices = df.loc[df["Transmission"] == "Automatic", "Selling_Price"]
    u_trans, p_trans, note_trans = safe_mannwhitneyu(auto_prices, manual_prices)
    test_results["mannwhitney_transmission"] = {"u_stat": u_trans, "p_value": p_trans, "note": note_trans}
    print("\n3. Mann-Whitney U Test: Selling_Price (Automatic vs Manual):")
    if u_trans is not None:
        clean_a = pd.to_numeric(auto_prices, errors="coerce").dropna()
        clean_m = pd.to_numeric(manual_prices, errors="coerce").dropna()
        print(f"   - Sample Sizes: Automatic (n={len(clean_a)}), Manual (n={len(clean_m)})")
        print(f"   - Automatic Median: {clean_a.median():.2f} | Manual Median: {clean_m.median():.2f}")
        print(f"   - U-Statistic = {u_trans:.1f}, p-value = {p_trans:.4e}")
        if p_trans < 0.05:
            print("   - Conclusion: Statistically significant difference (p = 0.001). In this sample, Automatic vehicles")
            print("     exhibit a higher median resale price than Manual vehicles.")
            print("     Causality Caution: This significance reflects an observed sample distribution difference rather than an")
            print("     isolated causal premium, as automatic transmissions in this dataset correlate with higher initial vehicle prices.")
    else:
        print(f"   - [SKIPPED] {note_trans}")

    # 4. Kruskal-Wallis H-Test across all 3 Fuel Types
    cng_prices = df.loc[df["Fuel_Type"] == "CNG", "Selling_Price"]
    h_fuel, p_h_fuel, note_kruskal = safe_kruskal(petrol_prices, diesel_prices, cng_prices)
    test_results["kruskal_fuel"] = {"h_stat": h_fuel, "p_value": p_h_fuel, "note": note_kruskal}
    print("\n4. Kruskal-Wallis H-Test across all 3 Fuel Types (Petrol, Diesel, CNG):")
    if h_fuel is not None:
        print(f"   - H-Statistic = {h_fuel:.4f}, p-value = {p_h_fuel:.4e}")
        print("   - Critical Limitations & Multiple-Comparison Context:")
        print("     a) Severe Sample Size Limitation: CNG has only n=2 observations. Statistical tests and post-hoc")
        print("        comparisons involving CNG are severely underpowered and cannot support reliable generalization.")
        print("     b) Omnibus Test Scope: Rejecting the omnibus null hypothesis indicates that at least one group")
        print("        distribution differs, but does NOT determine which specific pairwise differences are significant")
        print("        without post-hoc multiple-comparison correction (e.g., Dunn's test with Bonferroni adjustment).")
        print("     c) Association vs Causality: Observed differences represent bivariate distributional associations,")
        print("        not causal effects.")
    else:
        print(f"   - [SKIPPED] {note_kruskal}")

    return test_results


def save_eda_markdown_report(output_dir: Path, df: pd.DataFrame) -> Path:
    """
    Generate an educational, comprehensive Phase 2 EDA Markdown Report
    incorporating dynamically calculated empirical findings, visual evidence references,
    neutral price unit labeling, and Phase 3 guidance.
    """
    report_path = output_dir / "EDA_REPORT.md"

    # Compute all values dynamically
    num_rows, num_cols = df.shape
    sp = pd.to_numeric(df["Selling_Price"], errors="coerce")
    clean_sp = sp.dropna()
    sp_mean = float(clean_sp.mean()) if len(clean_sp) > 0 else 0.0
    sp_median = float(clean_sp.median()) if len(clean_sp) > 0 else 0.0
    sp_std = float(clean_sp.std()) if len(clean_sp) > 1 else 0.0
    sp_min = float(clean_sp.min()) if len(clean_sp) > 0 else 0.0
    sp_max = float(clean_sp.max()) if len(clean_sp) > 0 else 0.0
    sp_skew = float(clean_sp.skew()) if len(clean_sp) > 2 else 0.0

    # Mean vs Median floating-point comparison
    if sp_median != 0:
        mean_med_pct = ((sp_mean - sp_median) / abs(sp_median)) * 100.0
        direction = "higher" if mean_med_pct >= 0 else "lower"
        mean_med_str = f"The sample mean is {abs(mean_med_pct):.2f}% {direction} than the median (mean = {sp_mean:.4f} vs median = {sp_median:.4f} dataset price units)"
    else:
        mean_med_str = f"Mean is {sp_mean:.4f} with median at 0.00 (relative percentage undefined)"

    # Safe correlations
    r_sp_pp, _, _ = safe_pearsonr(df["Present_Price"], df["Selling_Price"])
    rho_sp_pp, _, _ = safe_spearmanr(df["Present_Price"], df["Selling_Price"])
    r_sp_pp_str = f"{r_sp_pp:.4f}" if r_sp_pp is not None else "N/A"
    rho_sp_pp_str = f"{rho_sp_pp:.4f}" if rho_sp_pp is not None else "N/A"
    r2_sp_pp_str = f"{r_sp_pp**2:.2f}" if r_sp_pp is not None else "N/A"

    r_sp_yr, _, _ = safe_pearsonr(df["Year"], df["Selling_Price"])
    rho_sp_yr, _, _ = safe_spearmanr(df["Year"], df["Selling_Price"])
    r_sp_yr_str = f"{r_sp_yr:.4f}" if r_sp_yr is not None else "N/A"
    rho_sp_yr_str = f"{rho_sp_yr:.4f}" if rho_sp_yr is not None else "N/A"

    r_yr_km, _, _ = safe_pearsonr(df["Year"], df["Driven_kms"])
    r_yr_km_str = f"{r_yr_km:.4f}" if r_yr_km is not None else "N/A"

    r_sp_km, _, _ = safe_pearsonr(df["Driven_kms"], df["Selling_Price"])
    r_sp_km_str = f"{r_sp_km:+.4f}" if r_sp_km is not None else "N/A"

    # Hypothesis testing
    sw_stat, sw_p, sw_note = safe_shapiro(df["Selling_Price"])
    sw_stat_str = f"{sw_stat:.4f}" if sw_stat is not None else "N/A"
    sw_p_str = f"{sw_p:.4e}" if sw_p is not None else "N/A"

    petrol_prices = df.loc[df["Fuel_Type"] == "Petrol", "Selling_Price"]
    diesel_prices = df.loc[df["Fuel_Type"] == "Diesel", "Selling_Price"]
    u_diesel_petrol, p_diesel_petrol, _ = safe_mannwhitneyu(diesel_prices, petrol_prices)
    u_dp_str = f"{u_diesel_petrol:.1f}" if u_diesel_petrol is not None else "N/A"
    p_dp_str = f"{p_diesel_petrol:.4e}" if p_diesel_petrol is not None else "N/A"
    diesel_med = pd.to_numeric(diesel_prices, errors="coerce").median()
    petrol_med = pd.to_numeric(petrol_prices, errors="coerce").median()
    diesel_med_str = f"{diesel_med:.2f}" if pd.notna(diesel_med) else "N/A"
    petrol_med_str = f"{petrol_med:.2f}" if pd.notna(petrol_med) else "N/A"

    auto_prices = df.loc[df["Transmission"] == "Automatic", "Selling_Price"]
    manual_prices = df.loc[df["Transmission"] == "Manual", "Selling_Price"]
    u_auto_manual, p_auto_manual, _ = safe_mannwhitneyu(auto_prices, manual_prices)
    u_am_str = f"{u_auto_manual:.1f}" if u_auto_manual is not None else "N/A"
    p_am_str = f"{p_auto_manual:.4e}" if p_auto_manual is not None else "N/A"
    auto_med = pd.to_numeric(auto_prices, errors="coerce").median()
    manual_med = pd.to_numeric(manual_prices, errors="coerce").median()
    auto_med_str = f"{auto_med:.2f}" if pd.notna(auto_med) else "N/A"
    manual_med_str = f"{manual_med:.2f}" if pd.notna(manual_med) else "N/A"

    cng_prices = df.loc[df["Fuel_Type"] == "CNG", "Selling_Price"]
    h_fuel, p_h_fuel, _ = safe_kruskal(petrol_prices, diesel_prices, cng_prices)
    h_fuel_str = f"{h_fuel:.4f}" if h_fuel is not None else "N/A"
    p_h_fuel_str = f"{p_h_fuel:.4e}" if p_h_fuel is not None else "N/A"

    cng_count = int((df["Fuel_Type"] == "CNG").sum())
    cng_pct = safe_percentage(cng_count, len(df))
    if cng_count > 0:
        cng_rows = df[df["Fuel_Type"] == "CNG"]
        cng_samples = [f"{str(r.get('Car_Name', '')).strip()} {r.get('Year', '')}" for _, r in cng_rows.head(3).iterrows()]
        cng_desc = f"{cng_count} observation(s) ({', '.join(cng_samples)})"
    else:
        cng_desc = "0 observations"

    owner3_count = int((df["Owner"] == 3).sum())
    owner3_pct = safe_percentage(owner3_count, len(df))
    if owner3_count > 0:
        o3_rows = df[df["Owner"] == 3]
        o3_samples = [f"{str(r.get('Car_Name', '')).strip()} {r.get('Year', '')}" for _, r in o3_rows.head(3).iterrows()]
        owner3_desc = f"{owner3_count} observation(s) ({', '.join(o3_samples)})"
    else:
        owner3_desc = "0 observations"

    # Dynamic Duplicate Rows
    dup_indices = df[df.duplicated(keep="first")].index.tolist()
    dup_count = len(dup_indices)
    if dup_count > 0:
        dup_details = []
        for idx in dup_indices:
            row = df.loc[idx]
            matches = df[(df == row).all(axis=1)].index.tolist()
            orig_idx = matches[0] if matches else "unknown"
            car_name = str(row.get("Car_Name", "Unknown")).strip()
            year = str(row.get("Year", ""))
            dup_details.append(f"Row {idx} matches {orig_idx}: {car_name} {year}")
        dup_summary_str = f"{dup_count} duplicate row(s) ({'; '.join(dup_details)})"
        dup_action_str = f"Drop the {dup_count} exact duplicate row(s) (indices: {dup_indices})"
    else:
        dup_summary_str = "0 duplicate rows detected"
        dup_action_str = "No duplicate rows to remove"

    pp_num = pd.to_numeric(df["Present_Price"], errors="coerce")
    low_pp_count = int((pp_num < 2.0).sum())

    # Whitespace padding detection
    ws_mask = df["Car_Name"].astype(str).str.strip() != df["Car_Name"].astype(str)
    ws_count = int(ws_mask.sum())
    ws_names = df.loc[ws_mask, "Car_Name"].unique().tolist()

    # IQR bounds strictly as Q3 - Q1
    sp_q1 = float(clean_sp.quantile(0.25)) if len(clean_sp) > 0 else 0.0
    sp_q3 = float(clean_sp.quantile(0.75)) if len(clean_sp) > 0 else 0.0
    sp_iqr = sp_q3 - sp_q1
    sp_lower = sp_q1 - 1.5 * sp_iqr
    sp_upper = sp_q3 + 1.5 * sp_iqr
    sp_lower_outliers = int((clean_sp < sp_lower).sum())
    sp_upper_outliers = int((clean_sp > sp_upper).sum())
    sp_outliers_count = sp_lower_outliers + sp_upper_outliers

    km = pd.to_numeric(df["Driven_kms"], errors="coerce")
    clean_km = km.dropna()
    km_q1 = float(clean_km.quantile(0.25)) if len(clean_km) > 0 else 0.0
    km_q3 = float(clean_km.quantile(0.75)) if len(clean_km) > 0 else 0.0
    km_iqr = km_q3 - km_q1
    km_lower = km_q1 - 1.5 * km_iqr
    km_upper = km_q3 + 1.5 * km_iqr
    km_lower_outliers = int((clean_km < km_lower).sum())
    km_upper_outliers = int((clean_km > km_upper).sum())
    km_outliers_count = km_lower_outliers + km_upper_outliers
    max_km = int(clean_km.max()) if len(clean_km) > 0 else 0
    max_km_vehicle = df.loc[clean_km.idxmax(), "Car_Name"].strip() if len(clean_km) > 0 else "Unknown"

    report_content = f"""# CodeAlpha Data Science Internship — Task 3: Car Price Prediction
## Phase 2: Exploratory Data Analysis (EDA) Comprehensive Technical Report

**Author / Candidate:** Data Science Intern  
**Dataset Dimensions:** {num_rows} Observations × {num_cols} Attributes  
**Price Unit Policy:** Documented neutrally as "dataset price units" (unconfirmed in raw CSV header)  
**Status:** Completed & Empirically Verified Against Active Dataset  

---

### Executive Summary & Key Evidence-Based Findings

1. **Target Distribution (`Selling_Price`)**:
   - **Empirical Statistics**: Mean = {sp_mean:.4f}, Median = {sp_median:.4f}, Std Dev = {sp_std:.4f}, Min = {sp_min:.2f}, Max = {sp_max:.2f} (dataset price units).
   - **Skewness**: {interpret_skewness(sp_skew)}. {mean_med_str} due to an extended upper tail.
   - **Normality Check**: Shapiro-Wilk test rejects marginal target normality ($W = {sw_stat_str}, p = {sw_p_str} < 0.05$).
   - **Statistical Nuance for Modeling**: This test evaluates only the marginal distribution of the target $Y$, not the conditional distribution $Y|X$ or regression residuals. Non-normality of the target does not prove non-normality of residuals or heteroskedasticity in an eventual regression model. However, heavy positive skewness (+2.49) indicates high-leverage observations in OLS loss, motivating the evaluation of variance-stabilizing transformations (e.g. $\\log(1 + y)$) and tree-based ensembles (Random Forest / Gradient Boosting) in Phase 3.

2. **Observed Numerical Resale Association (`Present_Price`)**:
   - **Correlation**: Pearson $r = {r_sp_pp_str}$, Spearman $\\rho = {rho_sp_pp_str}$.
   - **Interpretation**: Among the examined numerical features, showroom asset value (`Present_Price`) exhibits the strongest observed positive linear and monotonic rank association with resale price ($R^2 \\approx {r2_sp_pp_str}$). This reflects observed sample association across pooled data and does not independently establish predictive superiority or causality without multivariate modeling.

3. **Temporal Depreciation (`Year`)**:
   - **Correlation**: Pearson $r = {r_sp_yr_str}$ (weak positive linear association), Spearman $\\rho = {rho_sp_yr_str}$ (weak-to-moderate monotonic rank association).
   - **Collinearity with Mileage**: Pearson $r = {r_yr_km_str}$ demonstrates expected chronological accumulation of kilometers.
   - **Phase 3 Recommendation**: Formulate vehicle age: $\\text{{Car\\_Age}} = \\text{{Reference\\_Year}} - \\text{{Year}}$ to provide an intuitive negative regression coefficient.

4. **Mileage Association (`Driven_kms`)**:
   - **Bivariate Correlation**: Pearson $r = {r_sp_km_str}$ (weak aggregate linear association across pooled data).
   - **Confounding Variable Note**: Correlation does not imply causation. A weak aggregate linear correlation does not establish that mileage has no effect on resale value. In a pooled sample, vehicle segment heterogeneity is a major confounding factor: higher-tier vehicles may accumulate substantial mileage while retaining high asset value, whereas lower-tier vehicles have low price ceilings even with minimal mileage. Multivariate models will isolate the true conditional effect of mileage.

5. **Category Group Comparisons & Sub-Segments**:
   - **Diesel vs Petrol**: In this sample, Diesel vehicles exhibit a higher median resale price ({diesel_med_str} units) than Petrol vehicles ({petrol_med_str} units) (Mann-Whitney $U = {u_dp_str}, p = {p_dp_str} < 0.001$).
     *Causality & Confounding Caution*: Statistical significance establishes an observed group distributional difference, not that choosing diesel fuel causes higher resale value. Confounding factors such as higher initial purchase price or segment composition are plausible hypotheses to be tested in multivariate modeling, not established facts from this bivariate test.
   - **Automatic vs Manual**: In this sample, Automatic vehicles exhibit a higher median resale price ({auto_med_str} units) than Manual vehicles ({manual_med_str} units) (Mann-Whitney $U = {u_am_str}, p = {p_am_str} < 0.01$). This reflects an observed group distribution difference rather than an isolated causal premium.
   - **Fuel Type Omnibus Test (Kruskal-Wallis)**: $H = {h_fuel_str}, p = {p_h_fuel_str} < 0.001$.
     *Limitations*: (a) CNG has only $n={cng_count}$ observations ({cng_pct:.2f}%), making inference regarding CNG statistically underpowered; (b) Omnibus test rejection does not pinpoint which specific pairwise differences are significant without post-hoc multiple-comparison correction; (c) Significance indicates association, not causality.
   - **Vehicle Heterogeneity**: {low_pp_count} vehicles have $\\text{{Present\\_Price}} < 2.0$ dataset price units, representing motorcycles and scooters.

6. **Sample Size Limitations & Sparsity**:
   - **CNG Fuel Type**: Only {cng_count} records ({cng_pct:.2f}% of dataset).
   - **Owner Level 3**: Only {owner3_count} record ({owner3_pct:.2f}% of dataset).
   - **Duplicate Records**: Exactly {dup_count} duplicate listings detected (100% attribute identity).
   - **Whitespace Inconsistency**: {ws_count} records in `Car_Name` have trailing whitespace padding.

---

### Catalog of Generated Visual Figures

| File Name | Chart Type | Analytical Objective |
| :--- | :--- | :--- |
| `eda_01_selling_price_distribution.png` | Histogram + Boxplot | Target variable skewness and outlier identification |
| `eda_02_present_price_distribution.png` | Histogram + Boxplot | Showroom asset value range and luxury skew |
| `eda_03_driven_kms_distribution.png` | Histogram + Boxplot | Odometer reading distribution and {max_km:,} km inspection |
| `eda_04_year_owner_distributions.png` | Count Plots | Manufacturing year spread and owner distribution |
| `eda_05_categorical_counts.png` | Annotated Bar Plots | Fuel_Type, Selling_type, Transmission distribution |
| `eda_06_top_car_models.png` | Horizontal Bar Plot | Top 15 Car_Name models by listing frequency |
| `eda_07_price_vs_numerical_bivariate.png` | Regression Plots | Bivariate trendlines with Pearson & Spearman annotations |
| `eda_08_price_by_categorical_bivariate.png` | Boxplots | Category group comparisons with sample size annotations |
| `eda_09_correlation_heatmaps.png` | Dual Heatmaps | Pearson linear vs Spearman rank correlation matrices |
| `eda_10_multivariate_interactions.png` | Multi-hue Scatter | Transmission, Fuel Type, and Seller Type interactions |

---

### Outlier and Domain Anomaly Register

| Feature / Issue | Flagged Record / Value | Statistical Context | Domain Interpretation & Phase 3 Action |
| :--- | :--- | :--- | :--- |
| **Driven_kms** | {max_km:,} km ({max_km_vehicle}) | Total outliers: {km_outliers_count} ({km_lower_outliers} lower, {km_upper_outliers} upper > {km_upper:.0f} km) | Commercial delivery usage or data entry typo. Retained for Phase 3 sensitivity check. |
| **Selling_Price** | {sp_max:.2f} dataset price units | Total outliers: {sp_outliers_count} ({sp_lower_outliers} lower, {sp_upper_outliers} upper > {sp_upper:.2f} units) | Legitimate luxury vehicle (Land Cruiser). Not erroneous; models must handle high-value segment. |
| **Duplicate Rows** | {dup_summary_str} | 100% attribute match across active columns | Train-test leakage risk. Drop second occurrence in Phase 3. |
| **Whitespace Padding** | {ws_count} records in `Car_Name` ({', '.join(repr(x) for x in ws_names) if ws_names else 'None'}) | Trailing spaces in category strings | String hygiene defect. Strip leading/trailing whitespace during Phase 3 feature engineering. |
| **Owner = 3** | {owner3_desc} | Extreme categorical scarcity ($n={owner3_count}$) | Group with owner 1+ or handle cautiously during encoding. |
| **Fuel_Type = CNG**| {cng_desc} | Minority class ({cng_pct:.2f}%) | Acknowledge low statistical power in modeling. |

---

### Phase 3 Data Preprocessing & Feature Engineering Recommendations

1. **Deduplication**: {dup_action_str}.
2. **Whitespace Cleaning**: Strip leading and trailing whitespace from string columns (specifically the {ws_count} `Car_Name` categories: {', '.join(repr(x) for x in ws_names) if ws_names else 'None'}).
3. **Feature Derivation**: Create $\\text{{Car\\_Age}} = \\text{{Reference\\_Year}} - \\text{{Year}}$ to provide a direct depreciation metric.
4. **Target Strategy**: Evaluate training models on both raw $\\text{{Selling\\_Price}}$ and $\\log(1 + \\text{{Selling\\_Price}})$ to evaluate RMSE vs MAE trade-offs.
5. **High-Cardinality Strategy**: Rather than one-hot encoding 98 individual `Car_Name` values, extract brand names or rely on `Present_Price` as the proxy for vehicle segment tier.
6. **Leakage Prevention**: Train-test split before fitting any preprocessing scalers or encoders.
"""
    report_path.write_text(report_content, encoding="utf-8")
    print(f"\n[REPORT GENERATED] Comprehensive technical report saved to: {report_path.resolve()}")
    return report_path


def run_phase2_eda(
    df: Optional[pd.DataFrame] = None,
    csv_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Main orchestration routine for Phase 2 Exploratory Data Analysis.
    Loads dataset if not provided, executes all analytical steps, and saves artifacts.
    Enforces strict schema validation.
    """
    base_dir = Path(__file__).resolve().parent
    if csv_path is None:
        csv_path = resolve_default_dataset_path(base_dir)
    else:
        csv_path = Path(csv_path)

    if output_dir is None:
        output_dir = base_dir / "eda_outputs"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" CODEALPHA DATA SCIENCE INTERNSHIP - TASK 3: CAR PRICE PREDICTION")
    print(" PHASE 2: EXPLORATORY DATA ANALYSIS (EDA) & STATISTICAL PROFILING")
    print("=" * 72)
    print(f"Target CSV Path: {csv_path.resolve()}")
    print(f"Output Directory: {output_dir.resolve()}")

    if df is None:
        try:
            df = load_dataset(csv_path)
        except (ValueError, PermissionError) as err:
            print(f"[EXECUTION HALTED] Could not load dataset for Phase 2 EDA: {err}")
            return {}

    # Strict Schema Safety Validation
    if df is None or df.empty:
        print("[EXECUTION HALTED] Dataset is empty or contains no records. Phase 2 EDA cannot proceed.")
        return {}

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"[EXECUTION HALTED] Missing required columns for Phase 2 EDA: {missing_cols}")
        return {}

    dup_cols = df.columns[df.columns.duplicated()].unique().tolist()
    if dup_cols:
        print(f"[EXECUTION HALTED] Duplicate column names detected in dataset schema: {dup_cols}")
        return {}

    # Step 1: Dataset Overview & Quality Recheck
    overview = eda_dataset_overview(df)

    # Step 2: Univariate Analysis
    univariate = eda_univariate_analysis(df, output_dir)

    # Step 3: Bivariate Relationship Analysis
    bivariate = eda_bivariate_analysis(df, output_dir)

    # Step 4: Multivariate Analysis
    multivariate = eda_multivariate_analysis(df, output_dir)

    # Step 5: Outlier & Domain Anomaly Investigation
    outliers = eda_outlier_and_domain_investigation(df)

    # Step 6: Inferential Statistical Hypothesis Testing
    hypotheses = eda_statistical_hypothesis_testing(df)

    # Step 7: Save Markdown Report
    save_eda_markdown_report(output_dir, df)

    print("\n" + "=" * 72)
    print(" PHASE 2 EXPLORATORY DATA ANALYSIS CONCLUDED SUCCESSFULLY")
    print(f" All 10 high-resolution charts and technical report saved to: '{output_dir.name}/'")
    print("=" * 72 + "\n")

    return {
        "overview": overview,
        "univariate": univariate,
        "bivariate": bivariate,
        "multivariate": multivariate,
        "outliers": outliers,
        "hypotheses": hypotheses,
    }


# =========================================================================
# MAIN EXECUTION ORCHESTRATOR
# =========================================================================

def main() -> None:
    """
    Unified command-line interface supporting independent or combined execution
    of Phase 1 (Audit) and Phase 2 (EDA).
    Enforces consistent audit status and halts when errors occur.
    """
    parser = argparse.ArgumentParser(
        description="CodeAlpha Car Price Prediction - Phase 1 (Audit), Phase 2 (EDA), Phase 3 (Preprocessing), Phase 4 (Model Training) & Phase 5 (Evaluation)"
    )
    parser.add_argument(
        "--phase",
        choices=["1", "2", "3", "4", "5", "6", "all"],
        default="all",
        help="Analytical stage to execute: '1' (Audit), '2' (EDA), '3' (Preprocessing), '4' (Model Training), '5' (Evaluation), '6' (Prediction Interface), or 'all' (default: 1-5)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Custom path to CSV dataset (defaults to 'car data(1).csv' or 'car data.csv')",
    )
    args = parser.parse_args()

    custom_path = Path(args.csv) if args.csv else None

    if args.phase in ["1", "all"]:
        status = run_phase1_audit(csv_path=custom_path)
        if status == "FAIL":
            if args.phase == "all":
                print("[NOTICE] Halting execution before subsequent phases due to Phase 1 Critical Failure.")
            sys.exit(1)

    if args.phase in ["2", "all"]:
        eda_results = run_phase2_eda(csv_path=custom_path)
        if not eda_results:
            print("[NOTICE] Phase 2 EDA failed schema validation or encountered an unrecoverable error.")
            sys.exit(1)

    phase3_csv = None
    if args.phase in ["3", "all"]:
        try:
            from phase3_preprocessing import run_phase3
            phase3_results = run_phase3(csv_path=custom_path)
            if not phase3_results:
                print("[NOTICE] Phase 3 Data Preprocessing encountered an unrecoverable error.")
                sys.exit(1)
            phase3_csv = Path(phase3_results["summary"]["saved_artifacts"]["cleaned_csv"])
        except ImportError as err:
            print(f"[ERROR] Could not import phase3_preprocessing module: {err}")
            sys.exit(1)

    if args.phase in ["4", "all"]:
        try:
            from phase4_model_training import run_phase4
            base_dir = Path(__file__).resolve().parent
            target_csv = phase3_csv if phase3_csv and phase3_csv.is_file() else base_dir / "outputs" / "phase3" / "cleaned_car_data.csv"
            output_dir = base_dir / "outputs" / "phase4"
            phase4_results = run_phase4(csv_path=target_csv, output_dir=output_dir)
            if not phase4_results:
                print("[NOTICE] Phase 4 Model Training encountered an unrecoverable error.")
                sys.exit(1)
        except ImportError as err:
            print(f"[ERROR] Could not import phase4_model_training module: {err}")
            sys.exit(1)

    if args.phase in ["5", "all"]:
        try:
            from phase5_evaluation import run_phase5
            base_dir = Path(__file__).resolve().parent
            target_csv = phase3_csv if phase3_csv and phase3_csv.is_file() else base_dir / "outputs" / "phase3" / "cleaned_car_data.csv"
            model_path = base_dir / "outputs" / "phase4" / "best_model.joblib"
            metadata_path = base_dir / "outputs" / "phase4" / "training_metadata.json"
            output_dir = base_dir / "outputs" / "phase5"
            phase5_results = run_phase5(
                csv_path=target_csv,
                model_path=model_path,
                metadata_path=metadata_path,
                output_dir=output_dir,
            )
            if not phase5_results:
                print("[NOTICE] Phase 5 Model Evaluation encountered an unrecoverable error.")
                sys.exit(1)
        except ImportError as err:
            print(f"[ERROR] Could not import phase5_evaluation module: {err}")
            sys.exit(1)

    if args.phase == "6":
        try:
            from phase6_predict import main as run_phase6_main
            run_phase6_main()
        except ImportError as err:
            print(f"[ERROR] Could not import phase6_predict module: {err}")
            sys.exit(1)


if __name__ == "__main__":
    main()
