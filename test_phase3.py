"""
Comprehensive Validation & Test Suite for Phase 3 Data Preprocessing & Feature Engineering
Internship: CodeAlpha Data Science Internship - Task 3: Car Price Prediction
"""

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from phase3_preprocessing import (
    EXPECTED_COLUMNS,
    NUMERICAL_PREDICTORS,
    CATEGORICAL_PREDICTORS,
    EXCLUDED_PREDICTORS,
    TARGET_COLUMN,
    build_preprocessing_pipeline,
    clean_and_preprocess_dataset,
    resolve_default_dataset_path,
    run_phase3,
    validate_input_schema,
)


class TestPhase3Preprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent
        cls.raw_csv_path = resolve_default_dataset_path(cls.base_dir)
        cls.raw_df = pd.read_csv(cls.raw_csv_path)
        cls.raw_row_count = len(cls.raw_df)
        cls.raw_col_count = len(cls.raw_df.columns)

    def test_01_raw_dataset_integrity(self):
        """Ensure raw CSV has 301 rows and 9 columns and remains completely untouched."""
        self.assertEqual(self.raw_row_count, 301)
        self.assertEqual(self.raw_col_count, 9)
        self.assertEqual(list(self.raw_df.columns), EXPECTED_COLUMNS)

    def test_02_schema_validation_success(self):
        """Valid dataframe passes schema check."""
        valid, errors = validate_input_schema(self.raw_df)
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)

    def test_03_schema_validation_failures(self):
        """Missing columns or duplicate headers are caught cleanly."""
        # Empty dataframe
        valid, errors = validate_input_schema(pd.DataFrame())
        self.assertFalse(valid)
        self.assertIn("empty", errors[0].lower())

        # Missing required column
        bad_df = self.raw_df.drop(columns=["Selling_Price"])
        valid, errors = validate_input_schema(bad_df)
        self.assertFalse(valid)
        self.assertIn("Selling_Price", errors[0])

        # Duplicate column headers
        dup_col_df = pd.DataFrame([[1, 2]], columns=["Year", "Year"])
        valid, errors = validate_input_schema(dup_col_df)
        self.assertFalse(valid)
        self.assertIn("Duplicate column headers", errors[1])

    def test_04_deduplication_and_shape(self):
        """Cleaned dataset drops exact duplicate rows (301 -> 299) without modifying raw dataset."""
        cleaned_df, summary = clean_and_preprocess_dataset(self.raw_df)
        self.assertEqual(summary["initial_row_count"], 301)
        self.assertEqual(summary["deduplication"]["duplicates_removed_count"], 2)
        self.assertEqual(summary["final_row_count"], 299)
        self.assertEqual(len(cleaned_df), 299)
        self.assertEqual(int(cleaned_df.duplicated().sum()), 0)
        # Verify raw dataframe was not mutated
        self.assertEqual(len(self.raw_df), 301)

    def test_05_whitespace_normalization(self):
        """Whitespace in categorical text columns is trimmed."""
        cleaned_df, summary = clean_and_preprocess_dataset(self.raw_df)
        # Check Car_Name has no leading/trailing whitespace
        for val in cleaned_df["Car_Name"]:
            self.assertEqual(val, val.strip())
        for val in cleaned_df["Fuel_Type"]:
            self.assertEqual(val, val.strip())

    def test_06_target_validity_and_exclusion(self):
        """Target 'Selling_Price' is strictly positive, finite, and excluded from predictors."""
        cleaned_df, summary = clean_and_preprocess_dataset(self.raw_df)
        target = cleaned_df[TARGET_COLUMN]
        self.assertTrue((target > 0).all())
        self.assertTrue(np.isfinite(target).all())
        self.assertEqual(target.isna().sum(), 0)

        # Invariant: Target must never be in numerical or categorical predictors
        self.assertNotIn(TARGET_COLUMN, NUMERICAL_PREDICTORS)
        self.assertNotIn(TARGET_COLUMN, CATEGORICAL_PREDICTORS)

    def test_07_car_age_feature_engineering(self):
        """'Car_Age' feature is computed transparently and is non-negative."""
        ref_year = 2018
        cleaned_df, summary = clean_and_preprocess_dataset(self.raw_df, reference_year=ref_year)
        self.assertIn("Car_Age", cleaned_df.columns)
        self.assertTrue((cleaned_df["Car_Age"] >= 0).all())
        expected_age = ref_year - cleaned_df["Year"]
        pd.testing.assert_series_equal(cleaned_df["Car_Age"], expected_age, check_names=False)

    def test_08_car_name_exclusion_from_baseline(self):
        """'Car_Name' is retained in cleaned data for auditability but excluded from baseline predictors."""
        cleaned_df, summary = clean_and_preprocess_dataset(self.raw_df)
        self.assertIn("Car_Name", cleaned_df.columns)
        self.assertIn("Car_Name", summary["feature_architecture"]["excluded_from_baseline"])
        self.assertNotIn("Car_Name", NUMERICAL_PREDICTORS)
        self.assertNotIn("Car_Name", CATEGORICAL_PREDICTORS)

    def test_09_anti_leakage_pipeline_blueprint(self):
        """ColumnTransformer is configured properly with OneHotEncoder(drop='first') and is unfitted."""
        preprocessor = build_preprocessing_pipeline()
        self.assertIsInstance(preprocessor, ColumnTransformer)
        # Confirm it is NOT fitted yet (raises NotFittedError if transform called before fit)
        dummy_X = pd.DataFrame({
            "Present_Price": [10.0, 5.0],
            "Driven_kms": [20000, 40000],
            "Car_Age": [2, 5],
            "Owner": [0, 1],
            "Fuel_Type": ["Petrol", "Diesel"],
            "Selling_type": ["Dealer", "Individual"],
            "Transmission": ["Manual", "Automatic"],
        })
        with self.assertRaises(Exception):
            preprocessor.transform(dummy_X)

    def test_10_domain_sanity_rejections(self):
        """Non-positive target, inverted prices, or impossible years raise actionable errors."""
        # Non-positive selling price
        dirty_df1 = self.raw_df.copy()
        dirty_df1.loc[0, "Selling_Price"] = -1.0
        with self.assertRaises(ValueError) as ctx1:
            clean_and_preprocess_dataset(dirty_df1)
        self.assertIn("non-positive", str(ctx1.exception).lower())

        # Inverted price (Selling_Price > Present_Price)
        dirty_df2 = self.raw_df.copy()
        dirty_df2.loc[0, "Selling_Price"] = 999.0
        dirty_df2.loc[0, "Present_Price"] = 10.0
        with self.assertRaises(ValueError) as ctx2:
            clean_and_preprocess_dataset(dirty_df2)
        self.assertIn("selling_price > present_price", str(ctx2.exception).lower())

        # Year out of bounds
        dirty_df3 = self.raw_df.copy()
        dirty_df3.loc[0, "Year"] = 1850
        with self.assertRaises(ValueError) as ctx3:
            clean_and_preprocess_dataset(dirty_df3)
        self.assertIn("year", str(ctx3.exception).lower())

    def test_11_end_to_end_artifact_generation(self):
        """run_phase3 generates cleaned CSV and JSON summary matching expectations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            results = run_phase3(csv_path=self.raw_csv_path, output_dir=out_dir)
            self.assertTrue(bool(results))

            cleaned_csv = out_dir / "cleaned_car_data.csv"
            summary_json = out_dir / "preprocessing_summary.json"

            self.assertTrue(cleaned_csv.exists())
            self.assertTrue(summary_json.exists())

            df_out = pd.read_csv(cleaned_csv)
            self.assertEqual(len(df_out), 299)
            self.assertEqual(len(df_out.columns), 10)

            with open(summary_json, "r", encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["final_row_count"], 299)
            self.assertEqual(doc["deduplication"]["duplicates_removed_count"], 2)


if __name__ == "__main__":
    unittest.main()
