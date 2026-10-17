"""
Comprehensive Validation & Test Suite for Phase 6 Prediction Interface
Internship: CodeAlpha Data Science Internship - Task 3: Car Price Prediction
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from phase6_predict import (
    CATEGORICAL_FEATURES,
    FEATURES,
    HISTORICAL_REFERENCE_YEAR,
    NUMERIC_FEATURES,
    TARGET,
    get_supported_categories,
    get_supported_owners,
    load_artifacts,
    load_phase5_metrics,
    predict_price,
    validate_single_record,
    generate_phase6_report,
)


class TestPhase6Predict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent
        cls.model_path = cls.base_dir / "outputs" / "phase4" / "best_model.joblib"
        cls.metadata_path = cls.base_dir / "outputs" / "phase4" / "training_metadata.json"
        cls.csv_path = cls.base_dir / "outputs" / "phase3" / "cleaned_car_data.csv"

        cls.assertTrue(cls.model_path.is_file(), "Phase 4 model must exist.")
        cls.assertTrue(cls.metadata_path.is_file(), "Phase 4 metadata must exist.")
        cls.assertTrue(cls.csv_path.is_file(), "Phase 3 cleaned CSV must exist.")

        cls.model, cls.metadata, cls.df = load_artifacts(
            cls.model_path, cls.metadata_path, cls.csv_path
        )
        cls.categories = get_supported_categories(cls.df)
        cls.owners = get_supported_owners(cls.df)

    def test_01_artifacts_exist_and_load(self):
        """Check all required artifacts load cleanly and schema aligns."""
        self.assertTrue(hasattr(self.model, "predict"))
        self.assertEqual(self.metadata.get("target"), TARGET)
        self.assertEqual(self.metadata.get("features"), FEATURES)
        self.assertEqual(self.metadata.get("reference_year"), HISTORICAL_REFERENCE_YEAR)

    def test_02_supported_domain_values(self):
        """Verify supported categories and owner counts match cleaned dataset."""
        self.assertEqual(self.owners, [0, 1, 3])
        self.assertEqual(self.categories["Fuel_Type"], ["CNG", "Diesel", "Petrol"])
        self.assertEqual(self.categories["Selling_type"], ["Dealer", "Individual"])
        self.assertEqual(self.categories["Transmission"], ["Automatic", "Manual"])

    def test_03_validate_single_record_success(self):
        """Verify that a valid dictionary transforms into a 1-row DataFrame with exact feature order."""
        valid_input = {
            "Present_Price": 5.59,
            "Driven_kms": 27000,
            "Car_Age": 4,
            "Owner": 0,
            "Fuel_Type": "Petrol",
            "Selling_type": "Dealer",
            "Transmission": "Manual",
        }
        df_row = validate_single_record(valid_input, self.categories, self.owners)
        self.assertEqual(len(df_row), 1)
        self.assertListEqual(list(df_row.columns), FEATURES)
        self.assertEqual(df_row.loc[0, "Present_Price"], 5.59)
        self.assertEqual(df_row.loc[0, "Driven_kms"], 27000.0)
        self.assertEqual(df_row.loc[0, "Car_Age"], 4)
        self.assertEqual(df_row.loc[0, "Owner"], 0)
        self.assertEqual(df_row.loc[0, "Fuel_Type"], "Petrol")
        self.assertEqual(df_row.loc[0, "Selling_type"], "Dealer")
        self.assertEqual(df_row.loc[0, "Transmission"], "Manual")

    def test_04_validate_single_record_rejections(self):
        """Verify that out-of-domain, non-finite, and negative inputs are rejected."""
        base_valid = {
            "Present_Price": 5.59,
            "Driven_kms": 27000,
            "Car_Age": 4,
            "Owner": 0,
            "Fuel_Type": "Petrol",
            "Selling_type": "Dealer",
            "Transmission": "Manual",
        }

        # 1. Non-positive Present_Price
        bad_price = dict(base_valid, Present_Price=0.0)
        with self.assertRaises(ValueError):
            validate_single_record(bad_price, self.categories, self.owners)

        bad_price_neg = dict(base_valid, Present_Price=-3.5)
        with self.assertRaises(ValueError):
            validate_single_record(bad_price_neg, self.categories, self.owners)

        # 2. Negative Driven_kms
        bad_kms = dict(base_valid, Driven_kms=-500)
        with self.assertRaises(ValueError):
            validate_single_record(bad_kms, self.categories, self.owners)

        # 3. Negative Car_Age
        bad_age = dict(base_valid, Car_Age=-1)
        with self.assertRaises(ValueError):
            validate_single_record(bad_age, self.categories, self.owners)

        # 4. Invalid Owner (2 is not in [0, 1, 3])
        bad_owner = dict(base_valid, Owner=2)
        with self.assertRaises(ValueError):
            validate_single_record(bad_owner, self.categories, self.owners)

        # 5. Invalid Categories
        bad_fuel = dict(base_valid, Fuel_Type="Electric")
        with self.assertRaises(ValueError):
            validate_single_record(bad_fuel, self.categories, self.owners)

        bad_selling = dict(base_valid, Selling_type="Auction")
        with self.assertRaises(ValueError):
            validate_single_record(bad_selling, self.categories, self.owners)

        bad_trans = dict(base_valid, Transmission="CVT")
        with self.assertRaises(ValueError):
            validate_single_record(bad_trans, self.categories, self.owners)

        # 6. Non-finite values
        bad_nan = dict(base_valid, Present_Price=float("nan"))
        with self.assertRaises(ValueError):
            validate_single_record(bad_nan, self.categories, self.owners)

    def test_05_predict_price_inference(self):
        """Verify forward inference produces finite, deterministic prediction."""
        valid_input = {
            "Present_Price": 5.59,
            "Driven_kms": 27000,
            "Car_Age": 4,
            "Owner": 0,
            "Fuel_Type": "Petrol",
            "Selling_type": "Dealer",
            "Transmission": "Manual",
        }
        df_row = validate_single_record(valid_input, self.categories, self.owners)
        pred1 = predict_price(self.model, df_row)
        pred2 = predict_price(self.model, df_row)

        self.assertTrue(np.isfinite(pred1))
        self.assertAlmostEqual(pred1, pred2, places=7)  # Deterministic inference allowing multi-threaded float summation
        self.assertAlmostEqual(pred1, 3.8400, places=3)

    def test_06_load_phase5_metrics(self):
        """Verify load_phase5_metrics correctly loads real metrics and returns None on missing directory."""
        phase6_dir = self.base_dir / "outputs" / "phase6"
        metrics = load_phase5_metrics(phase6_dir)
        self.assertIsNotNone(metrics)
        self.assertIn("RMSE", metrics)
        self.assertIn("MAE", metrics)
        self.assertAlmostEqual(metrics["RMSE"], 3.1668, places=3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_metrics = load_phase5_metrics(Path(tmp_dir) / "phase6")
            self.assertIsNone(missing_metrics)

    def test_07_estimator_metadata_mismatch(self):
        """Verify load_artifacts raises ValueError if metadata selected_model conflicts with pipeline estimator."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_meta_path = Path(tmp_dir) / "bad_metadata.json"
            bad_meta = dict(self.metadata)
            bad_meta["selected_model"] = "Gradient Boosting Regressor"
            with open(bad_meta_path, "w", encoding="utf-8") as f:
                json.dump(bad_meta, f)

            with self.assertRaises(ValueError) as ctx:
                load_artifacts(self.model_path, bad_meta_path, self.csv_path)
            self.assertIn("specifies 'Gradient Boosting Regressor'", str(ctx.exception))

    def test_08_report_generation_missing_medae_safe(self):
        """Verify report generation handles missing MedAE safely without crashing on NoneType formatting."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_dir = Path(tmp_dir) / "phase6"
            valid_input = {
                "Present_Price": 5.59,
                "Driven_kms": 27000,
                "Car_Age": 4,
                "Owner": 0,
                "Fuel_Type": "Petrol",
                "Selling_type": "Dealer",
                "Transmission": "Manual",
            }
            sample_df = validate_single_record(valid_input, self.categories, self.owners)

            # Phase 5 metrics dictionary with MedAE missing entirely (None)
            incomplete_metrics = {
                "RMSE": 3.1668,
                "MAE": 1.3567,
                "Median_Absolute_Error": None,
                "R2": 0.6109,
            }

            # Should not raise TypeError: unsupported format string passed to NoneType
            report_path = generate_phase6_report(
                report_dir=report_dir,
                metadata=self.metadata,
                phase5_metrics=incomplete_metrics,
                sample_row=sample_df,
                sample_prediction=3.84,
                supported_categories=self.categories,
                supported_owners=self.owners,
            )
            self.assertTrue(report_path.is_file())
            content = report_path.read_text(encoding="utf-8")
            self.assertIn("unavailable", content)
            self.assertIn("3.8400", content)


if __name__ == "__main__":
    unittest.main()
