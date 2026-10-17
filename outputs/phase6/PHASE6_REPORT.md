# Phase 6: Car Price Prediction Interface Report

**CodeAlpha Data Science Internship — Task 3: Car Price Prediction**

---

## 1. Interface & Model Artifact Overview

- **Model Pipeline Artifact**: `outputs/phase4/best_model.joblib`
- **Model Architecture**: `Random Forest` (encapsulated inside a scikit-learn `Pipeline` with `StandardScaler` and `OneHotEncoder`)
- **Execution Mode**: **Inference Only**. The saved pipeline is evaluated as a frozen artifact; no retraining, parameter modification, or refitting occurs.
- **Price Units**: Predictions are output strictly in **dataset price units** (the source dataset does not confirm a currency/unit).

---

## 2. Input Feature Schema & Validation Invariants

Input records are defensively validated against the exact 7 modeling features in training order:

| Feature Name | Type | Validation Invariants & Constraints | Supported Domain Values |
| :--- | :--- | :--- | :--- |
| `Present_Price` | Numeric (float) | Strictly positive (> 0), finite | Continuously valued price units |
| `Driven_kms` | Numeric (float) | Non-negative (>= 0), finite | Odometer kilometres |
| `Car_Age` | Integer | Non-negative (>= 0), finite integer | Years relative to reference year 2018 |
| `Owner` | Integer | Supported training-dataset values (observed categories) | `[0, 1, 3]` |
| `Fuel_Type` | Categorical (str) | Exact match against dataset categories | `['CNG', 'Diesel', 'Petrol']` |
| `Selling_type` | Categorical (str) | Exact match against dataset categories | `['Dealer', 'Individual']` |
| `Transmission` | Categorical (str) | Exact match against dataset categories | `['Automatic', 'Manual']` |

> **Domain Note on Features & Categories**:
> - Rejection of owner counts outside `[0, 1, 3]` reflects the domain observed in the training distribution, rather than implying other values are impossible in real-world scenarios.
> - The model's training distribution contains limited representation for high-owner counts and commercial vehicles.

---

## 3. Historical Reference Year Limitation

> **Important Domain Limitation on `Car_Age`:**
> In Phase 3 feature engineering, `Car_Age` was defined as:
> `Car_Age = 2018 - Year`
> Because 2018 is the maximum manufacturing year present in the dataset, `Car_Age` cannot be interpreted as age relative to the current calendar year. Inputting the age of a car relative to today's date would introduce severe temporal distribution shift and generate invalid estimates.

---

## 4. Phase 5 Holdout Performance Context

- **Holdout Test Set Size**: 60 records (20% held-out test partition)
- **Holdout Test RMSE**: `3.1668 dataset price units`
- **Holdout Test MAE**: `1.3567 dataset price units`
- **Holdout Test Median Absolute Error (MedAE)**: `0.4081 dataset price units`
- **Holdout Test R²**: `0.6109 (relative variance explained; not percentage accuracy)`

*Performance Observation*: For mainstream volume vehicles, the model predicts selling prices with a median absolute error of approximately 0.4081 dataset price units. High-present-price vehicles with large squared deviations elevate the overall RMSE.

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

---

## 6. Verified Prediction Execution

The following verified prediction was executed against the frozen Phase 4 pipeline:

| Input Feature | Value | Feature Description |
| :--- | :--- | :--- |
| `Present_Price` | `5.59` | Showroom price in dataset units |
| `Driven_kms` | `27000.0` | Total odometer distance |
| `Car_Age` | `4` | Relative to reference year 2018 |
| `Owner` | `0` | Previous owner category count |
| `Fuel_Type` | `Petrol` | Primary powertrain fuel |
| `Selling_type` | `Dealer` | Seller classification |
| `Transmission` | `Manual` | Gearbox transmission |

**Predicted Selling Price**: **`3.8400`** dataset price units

---

## 7. Disclaimers & Practical Scope

1. **Not a Guaranteed Market Quote**: The predicted selling price reflects historical relationships learned from the 299-record dataset and is not an appraisal, commercial valuation, or guaranteed dealer offer.
2. **Out-of-Distribution Inputs**: Vehicles with features beyond the training range (e.g., modern electric cars, vintage pre-2000 models, or post-pandemic inflation) are outside the model's operational domain.
