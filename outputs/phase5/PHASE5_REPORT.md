# Phase 5: Final Model Evaluation Report

**CodeAlpha Data Science Internship — Task 3: Car Price Prediction**

---

## 1. Evaluation Setup & Integrity Guarantees

- **Selected Model Pipeline**: `Random Forest` (fitted in Phase 4 on 239 training rows)
- **Model Artifact Checksum (SHA-256)**: `cd1d6160fba8288755b83a3b64e0ed3c3e33365705e4af615c92e27a11eccb3a`
- **Holdout Test Set Size**: 60 records (exactly 20.0% of the 299 cleaned records)
- **Evaluation Protocol**:
  - The Phase 5 script evaluates the saved Phase 4 champion model (`best_model.joblib`) strictly on the reserved holdout partition without retraining or tuning.
  - The partition was reconstructed using the recorded split settings (`test_size=0.20`, `random_state=42`).

---

## 2. Holdout Test Performance Metrics

| Metric | Holdout Test Result | Phase 4 CV Training Baseline | Metric Interpretation & Nature |
| :--- | :---: | :---: | :--- |
| **RMSE** (Root Mean Squared Error) | **3.1668** | 1.5766 | Standard deviation of residuals; heavily penalizes large errors. Lower is better. |
| **MAE** (Mean Absolute Error) | **1.3567** | 0.7352 | Mean absolute magnitude of errors across all test cars. Lower is better. |
| **Median Absolute Error** | **0.4081** | — | Median error magnitude; robust against high-leverage outliers. Lower is better. |
| **R²** (Coefficient of Determination) | **0.6109** | 0.8926 | Proportion of target variance explained relative to a baseline mean predictor. Higher is better. |

> **Crucial Statistical Note on R² and MAPE:**
> - $R^2$ measures explained variance, **not** percentage accuracy of individual car price estimates.
> - Mean Absolute Percentage Error (MAPE) is intentionally excluded from the primary metric suite because car selling prices in this dataset contain very low values (e.g., 0.1–0.5 units), where minor absolute deviations yield misleadingly inflated percentage errors.

### Generalization & Cross-Validation Gap Analysis

The holdout test RMSE (**3.1668** dataset price units) is materially higher than the training 5-fold cross-validation mean RMSE (**1.5766** dataset price units). Similarly, the holdout test $R^2$ (**0.6109**) is lower than the cross-validation mean $R^2$ (**0.8926**).

This performance divergence represents an important generalization concern:
- The gap indicates that the model experiences greater residual dispersion on fresh holdout records than observed across internal validation folds.
- **Important Statistical Precaution**: This gap alone does **not** conclusively prove severe model overfitting across the entire feature space. As demonstrated below, half the observations have absolute errors at or below **0.4081** units, subject to ties. Rather than a proven universal flaw, a plausible working hypothesis is that a few high-value records with large deviations disproportionately elevate RMSE due to error squaring.

---

## 3. Residual Diagnostics & Error Distribution (Residual = Actual - Predicted)

1. **Typical Performance (Median Absolute Error = 0.4081 units)**:
   The Median Absolute Error (0.4081 dataset price units) indicates that half the observations have absolute errors at or below 0.4081 units, subject to ties. It provides a robust measure of typical error without implying that all vehicles are accurately estimated.
2. **Observed Residual Disparity & RMSE (3.1668 units)**:
   Because squared error squares large residuals, a small number of vehicles with larger deviations disproportionately increase RMSE relative to MAE (1.3567) and MedAE (0.4081). This is presented as an observed empirical characteristic of the sample residuals, rather than a definitive causal proof of model pathology.
3. **Largest Residual Cases on the Holdout Partition (Dynamically Identified)**:
| Source Row Index | Actual Selling Price | Predicted Selling Price | Signed Error (Actual - Pred) | Absolute Error |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Row 84 | 2.50 | 16.25 | -13.75 | 13.75 |
| Row 92 | 4.00 | 15.65 | -11.65 | 11.65 |
| Row 77 | 5.25 | 16.46 | -11.21 | 11.21 |


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
