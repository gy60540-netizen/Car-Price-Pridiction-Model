"""
Comprehensive Validation & Test Suite for Phase 4 Model Training & Cross-Validation
Internship: CodeAlpha Data Science Internship - Task 3: Car Price Prediction
"""

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from phase4_model_training import (
    CATEGORICAL_FEATURES,
    FEATURES,
    N_SPLITS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    get_models,
    load_and_validate,
    make_pipeline,
    run_phase4,
)


class TestPhase4ModelTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent
        cls.phase3_csv = cls.base_dir / "outputs" / "phase3" / "cleaned_car_data.csv"
        cls.assertTrue(cls.phase3_csv.is_file(), "Phase 3 cleaned dataset must exist.")
        cls.df = pd.read_csv(cls.phase3_csv)

    def test_01_modeling_dataset_integrity(self):
        """Verify Phase 3 cleaned dataset has 299 rows and contains all required features + target."""
        self.assertEqual(len(self.df), 299)
        for col in FEATURES + [TARGET]:
            self.assertIn(col, self.df.columns)
        self.assertEqual(int(self.df.duplicated().sum()), 0)
        self.assertTrue((self.df[TARGET] > 0).all())

    def test_02_target_exclusion_invariant(self):
        """Invariant: Target 'Selling_Price' must not be in predictor lists."""
        self.assertNotIn(TARGET, FEATURES)
        self.assertNotIn(TARGET, NUMERIC_FEATURES)
        self.assertNotIn(TARGET, CATEGORICAL_FEATURES)

    def test_03_load_and_validate_success(self):
        """Valid dataframe loads without errors."""
        loaded_df = load_and_validate(self.phase3_csv)
        self.assertEqual(len(loaded_df), 299)

    def test_04_load_and_validate_rejections(self):
        """Defensive validation catches missing columns, non-finite values, and non-positive targets."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Missing column
            bad_df1 = self.df.drop(columns=["Present_Price"])
            p1 = tmp_path / "bad1.csv"
            bad_df1.to_csv(p1, index=False)
            with self.assertRaises(ValueError) as ctx1:
                load_and_validate(p1)
            self.assertIn("Present_Price", str(ctx1.exception))

            # Non-positive target
            bad_df2 = self.df.copy()
            bad_df2.loc[0, TARGET] = -2.5
            p2 = tmp_path / "bad2.csv"
            bad_df2.to_csv(p2, index=False)
            with self.assertRaises(ValueError) as ctx2:
                load_and_validate(p2)
            self.assertIn("strictly positive", str(ctx2.exception).lower())

            # Non-finite value in numeric column
            bad_df3 = self.df.copy()
            bad_df3.loc[0, "Driven_kms"] = np.nan
            p3 = tmp_path / "bad3.csv"
            bad_df3.to_csv(p3, index=False)
            with self.assertRaises(ValueError) as ctx3:
                load_and_validate(p3)
            self.assertIn("missing values", str(ctx3.exception).lower())

    def test_05_candidate_models_structure(self):
        """Ensure all 5 required regression algorithms are present."""
        models = get_models()
        expected = [
            "Linear Regression",
            "Ridge Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
        ]
        self.assertEqual(sorted(models.keys()), sorted(expected))

    def test_06_pipeline_construction(self):
        """make_pipeline builds a valid Pipeline with ColumnTransformer and estimator."""
        models = get_models()
        pipe = make_pipeline(models["Linear Regression"])
        self.assertIsInstance(pipe, Pipeline)
        self.assertIn("preprocessor", pipe.named_steps)
        self.assertIn("model", pipe.named_steps)

    def test_07_end_to_end_phase4_execution(self):
        """Execute run_phase4 in a temporary directory and verify all artifacts and metadata."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            results = run_phase4(csv_path=self.phase3_csv, output_dir=out_dir)

            self.assertIn("best_pipeline", results)
            self.assertIn("results_df", results)
            self.assertIn("selected_model", results)

            # Check files generated
            model_file = out_dir / "best_model.joblib"
            cv_file = out_dir / "cross_validation_results.csv"
            meta_file = out_dir / "training_metadata.json"

            self.assertTrue(model_file.is_file())
            self.assertTrue(cv_file.is_file())
            self.assertTrue(meta_file.is_file())

            # Verify CV results table
            cv_df = pd.read_csv(cv_file)
            self.assertEqual(len(cv_df), 5)
            self.assertEqual(list(cv_df.columns), [
                "model", "cv_rmse_mean", "cv_rmse_std",
                "cv_mae_mean", "cv_mae_std", "cv_r2_mean", "cv_r2_std"
            ])
            # Selected model must match top row (lowest RMSE)
            self.assertEqual(results["selected_model"], cv_df.iloc[0]["model"])
            self.assertEqual(cv_df.iloc[0]["model"], "Random Forest")

            # Verify saved model can be reloaded and predict
            reloaded = joblib.load(model_file)
            sample_X = self.df[FEATURES].iloc[:2]
            preds = reloaded.predict(sample_X)
            self.assertEqual(len(preds), 2)
            self.assertTrue(np.isfinite(preds).all())

            # Verify metadata JSON fields
            with meta_file.open(encoding="utf-8") as f:
                meta = json.load(f)

            self.assertEqual(meta["phase"], 4)
            self.assertEqual(meta["target"], TARGET)
            self.assertEqual(meta["features"], FEATURES)
            self.assertEqual(meta["random_state"], RANDOM_STATE)
            self.assertEqual(meta["test_size"], TEST_SIZE)
            self.assertEqual(meta["training_rows"], 239)
            self.assertEqual(meta["reserved_test_rows"], 60)
            self.assertEqual(meta["cross_validation"]["n_splits"], N_SPLITS)
            self.assertFalse(meta["test_partition_used_for_model_selection"])
            self.assertFalse(meta["test_partition_evaluated_in_phase4"])
            self.assertEqual(meta["selected_model"], "Random Forest")
            self.assertEqual(meta["reference_year"], 2018)


if __name__ == "__main__":
    unittest.main()
