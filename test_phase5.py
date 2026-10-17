"""
Comprehensive Validation & Test Suite for Phase 5 Holdout Model Evaluation
Internship: CodeAlpha Data Science Internship - Task 3: Car Price Prediction
"""

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from phase5_evaluation import (
    EXPECTED_TEST_ROWS,
    EXPECTED_TRAIN_ROWS,
    FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    load_inputs,
    reconstruct_holdout,
    run_phase5,
)


class TestPhase5Evaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent
        cls.phase3_csv = cls.base_dir / "outputs" / "phase3" / "cleaned_car_data.csv"
        cls.model_path = cls.base_dir / "outputs" / "phase4" / "best_model.joblib"
        cls.metadata_path = cls.base_dir / "outputs" / "phase4" / "training_metadata.json"
        cls.phase5_dir = cls.base_dir / "outputs" / "phase5"

        cls.assertTrue(cls.phase3_csv.is_file(), "Phase 3 cleaned dataset must exist.")
        cls.assertTrue(cls.model_path.is_file(), "Phase 4 model pipeline must exist.")
        cls.assertTrue(cls.metadata_path.is_file(), "Phase 4 metadata must exist.")

    def test_01_inputs_and_artifacts_exist(self):
        """Check all required inputs from Phase 3 and Phase 4 exist."""
        self.assertTrue(self.phase3_csv.is_file())
        self.assertTrue(self.model_path.is_file())
        self.assertTrue(self.metadata_path.is_file())

    def test_02_load_inputs_success(self):
        """Verify load_inputs cleanly loads Phase 3 CSV, fitted pipeline, and metadata."""
        df, model, metadata = load_inputs(self.phase3_csv, self.model_path, self.metadata_path)
        self.assertEqual(len(df), 299)
        self.assertTrue(hasattr(model, "predict"))
        self.assertEqual(metadata.get("selected_model"), "Random Forest")
        self.assertEqual(metadata.get("random_state"), 42)

    def test_03_reconstruct_holdout_exactness(self):
        """Verify holdout partition reconstruction matches Phase 4 sizes (239 train / 60 test)."""
        df, _, metadata = load_inputs(self.phase3_csv, self.model_path, self.metadata_path)
        X_test, y_test, source_indices = reconstruct_holdout(df, metadata)

        self.assertEqual(len(X_test), EXPECTED_TEST_ROWS)
        self.assertEqual(len(y_test), EXPECTED_TEST_ROWS)
        self.assertEqual(len(source_indices), EXPECTED_TEST_ROWS)
        self.assertListEqual(list(X_test.columns), FEATURES)
        self.assertEqual(y_test.name, TARGET)

    def test_04_prediction_shape_and_finite(self):
        """Verify that model predictions are finite and strictly match test partition shape."""
        df, model, metadata = load_inputs(self.phase3_csv, self.model_path, self.metadata_path)
        X_test, y_test, _ = reconstruct_holdout(df, metadata)

        preds = model.predict(X_test)
        self.assertEqual(preds.shape, (60,))
        self.assertTrue(np.isfinite(preds).all())

    def test_05_execution_and_artifacts_integrity(self):
        """Run Phase 5 in a temporary directory and verify all 5 expected artifacts are generated."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_output = Path(tmp_dir)
            preds_df, metrics_payload = run_phase5(
                csv_path=self.phase3_csv,
                model_path=self.model_path,
                metadata_path=self.metadata_path,
                output_dir=tmp_output,
            )

            # Check files exist
            pred_file = tmp_output / "test_predictions.csv"
            metrics_file = tmp_output / "evaluation_metrics.json"
            plot1 = tmp_output / "actual_vs_predicted.png"
            plot2 = tmp_output / "residual_diagnostics.png"
            report_file = tmp_output / "PHASE5_REPORT.md"

            self.assertTrue(pred_file.is_file())
            self.assertTrue(metrics_file.is_file())
            self.assertTrue(plot1.is_file())
            self.assertTrue(plot2.is_file())
            self.assertTrue(report_file.is_file())

            # Verify predictions integrity
            self.assertEqual(len(preds_df), 60)
            self.assertIn("source_row_index", preds_df.columns)
            self.assertIn("actual_selling_price", preds_df.columns)
            self.assertIn("predicted_selling_price", preds_df.columns)
            self.assertIn("error_actual_minus_predicted", preds_df.columns)
            self.assertIn("absolute_error", preds_df.columns)

            # Verify reported RMSE matches predictions RMSE
            saved_df = pd.read_csv(pred_file)
            computed_rmse = float(np.sqrt(mean_squared_error(
                saved_df["actual_selling_price"],
                saved_df["predicted_selling_price"],
            )))
            self.assertAlmostEqual(computed_rmse, metrics_payload["test_metrics"]["RMSE"], places=6)

    def test_06_defensive_validation_errors(self):
        """Ensure invalid configurations and corrupted files are defensively rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Missing file error
            with self.assertRaises(FileNotFoundError):
                load_inputs(tmp_path / "nonexistent.csv", self.model_path, self.metadata_path)

            # 2. Metadata seed mismatch
            bad_meta_path = tmp_path / "bad_metadata.json"
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["random_state"] = 999
            with open(bad_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)

            df, _, _ = load_inputs(self.phase3_csv, self.model_path, self.metadata_path)
            with self.assertRaises(ValueError):
                reconstruct_holdout(df, meta)

            # 3. Non-positive target rejection
            bad_df_path = tmp_path / "bad_df.csv"
            bad_df = pd.read_csv(self.phase3_csv)
            bad_df.loc[0, TARGET] = -1.0
            bad_df.to_csv(bad_df_path, index=False)
            with self.assertRaises(ValueError):
                load_inputs(bad_df_path, self.model_path, self.metadata_path)


if __name__ == "__main__":
    unittest.main()
