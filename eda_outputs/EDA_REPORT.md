# CodeAlpha Data Science Internship — Task 3: Car Price Prediction
## Phase 2: Exploratory Data Analysis (EDA) Comprehensive Technical Report

**Author / Candidate:** Data Science Intern  
**Dataset Dimensions:** 301 Observations × 9 Attributes  
**Price Unit Policy:** Documented neutrally as "dataset price units" (unconfirmed in raw CSV header)  
**Status:** Completed & Empirically Verified Against Active Dataset  

---

### Executive Summary & Key Evidence-Based Findings

1. **Target Distribution (`Selling_Price`)**:
   - **Empirical Statistics**: Mean = 4.6613, Median = 3.6000, Std Dev = 5.0828, Min = 0.10, Max = 35.00 (dataset price units).
   - **Skewness**: +2.4934 (Highly right-skewed / positive tail). The sample mean is 29.48% higher than the median (mean = 4.6613 vs median = 3.6000 dataset price units) due to an extended upper tail.
   - **Normality Check**: Shapiro-Wilk test rejects marginal target normality ($W = 0.7607, p = 1.2634e-20 < 0.05$).
   - **Statistical Nuance for Modeling**: This test evaluates only the marginal distribution of the target $Y$, not the conditional distribution $Y|X$ or regression residuals. Non-normality of the target does not prove non-normality of residuals or heteroskedasticity in an eventual regression model. However, heavy positive skewness (+2.49) indicates high-leverage observations in OLS loss, motivating the evaluation of variance-stabilizing transformations (e.g. $\log(1 + y)$) and tree-based ensembles (Random Forest / Gradient Boosting) in Phase 3.

2. **Observed Numerical Resale Association (`Present_Price`)**:
   - **Correlation**: Pearson $r = 0.8789$, Spearman $\rho = 0.9083$.
   - **Interpretation**: Among the examined numerical features, showroom asset value (`Present_Price`) exhibits the strongest observed positive linear and monotonic rank association with resale price ($R^2 \approx 0.77$). This reflects observed sample association across pooled data and does not independently establish predictive superiority or causality without multivariate modeling.

3. **Temporal Depreciation (`Year`)**:
   - **Correlation**: Pearson $r = 0.2361$ (weak positive linear association), Spearman $\rho = 0.2826$ (weak-to-moderate monotonic rank association).
   - **Collinearity with Mileage**: Pearson $r = -0.5243$ demonstrates expected chronological accumulation of kilometers.
   - **Phase 3 Recommendation**: Formulate vehicle age: $\text{Car\_Age} = \text{Reference\_Year} - \text{Year}$ to provide an intuitive negative regression coefficient.

4. **Mileage Association (`Driven_kms`)**:
   - **Bivariate Correlation**: Pearson $r = +0.0292$ (weak aggregate linear association across pooled data).
   - **Confounding Variable Note**: Correlation does not imply causation. A weak aggregate linear correlation does not establish that mileage has no effect on resale value. In a pooled sample, vehicle segment heterogeneity is a major confounding factor: higher-tier vehicles may accumulate substantial mileage while retaining high asset value, whereas lower-tier vehicles have low price ceilings even with minimal mileage. Multivariate models will isolate the true conditional effect of mileage.

5. **Category Group Comparisons & Sub-Segments**:
   - **Diesel vs Petrol**: In this sample, Diesel vehicles exhibit a higher median resale price (7.75 units) than Petrol vehicles (2.65 units) (Mann-Whitney $U = 12467.0, p = 9.0240e-19 < 0.001$).
     *Causality & Confounding Caution*: Statistical significance establishes an observed group distributional difference, not that choosing diesel fuel causes higher resale value. Confounding factors such as higher initial purchase price or segment composition are plausible hypotheses to be tested in multivariate modeling, not established facts from this bivariate test.
   - **Automatic vs Manual**: In this sample, Automatic vehicles exhibit a higher median resale price (5.80 units) than Manual vehicles (3.25 units) (Mann-Whitney $U = 6902.5, p = 1.0320e-03 < 0.01$). This reflects an observed group distribution difference rather than an isolated causal premium.
   - **Fuel Type Omnibus Test (Kruskal-Wallis)**: $H = 78.8111, p = 7.6983e-18 < 0.001$.
     *Limitations*: (a) CNG has only $n=2$ observations (0.66%), making inference regarding CNG statistically underpowered; (b) Omnibus test rejection does not pinpoint which specific pairwise differences are significant without post-hoc multiple-comparison correction; (c) Significance indicates association, not causality.
   - **Vehicle Heterogeneity**: 98 vehicles have $\text{Present\_Price} < 2.0$ dataset price units, representing motorcycles and scooters.

6. **Sample Size Limitations & Sparsity**:
   - **CNG Fuel Type**: Only 2 records (0.66% of dataset).
   - **Owner Level 3**: Only 1 record (0.33% of dataset).
   - **Duplicate Records**: Exactly 2 duplicate listings detected (100% attribute identity).
   - **Whitespace Inconsistency**: 5 records in `Car_Name` have trailing whitespace padding.

---

### Catalog of Generated Visual Figures

| File Name | Chart Type | Analytical Objective |
| :--- | :--- | :--- |
| `eda_01_selling_price_distribution.png` | Histogram + Boxplot | Target variable skewness and outlier identification |
| `eda_02_present_price_distribution.png` | Histogram + Boxplot | Showroom asset value range and luxury skew |
| `eda_03_driven_kms_distribution.png` | Histogram + Boxplot | Odometer reading distribution and 500,000 km inspection |
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
| **Driven_kms** | 500,000 km (Activa 3g) | Total outliers: 8 (0 lower, 8 upper > 99418 km) | Commercial delivery usage or data entry typo. Retained for Phase 3 sensitivity check. |
| **Selling_Price** | 35.00 dataset price units | Total outliers: 17 (0 lower, 17 upper > 13.65 units) | Legitimate luxury vehicle (Land Cruiser). Not erroneous; models must handle high-value segment. |
| **Duplicate Rows** | 2 duplicate row(s) (Row 17 matches 15: ertiga 2016; Row 93 matches 51: fortuner 2015) | 100% attribute match across active columns | Train-test leakage risk. Drop second occurrence in Phase 3. |
| **Whitespace Padding** | 5 records in `Car_Name` ('KTM 390 Duke ', 'Yamaha Fazer ', 'TVS Sport ', 'Honda Dream Yuga ', 'Yamaha FZ S ') | Trailing spaces in category strings | String hygiene defect. Strip leading/trailing whitespace during Phase 3 feature engineering. |
| **Owner = 3** | 1 observation(s) (camry 2006) | Extreme categorical scarcity ($n=1$) | Group with owner 1+ or handle cautiously during encoding. |
| **Fuel_Type = CNG**| 2 observation(s) (wagon r 2015, sx4 2011) | Minority class (0.66%) | Acknowledge low statistical power in modeling. |

---

### Phase 3 Data Preprocessing & Feature Engineering Recommendations

1. **Deduplication**: Drop the 2 exact duplicate row(s) (indices: [17, 93]).
2. **Whitespace Cleaning**: Strip leading and trailing whitespace from string columns (specifically the 5 `Car_Name` categories: 'KTM 390 Duke ', 'Yamaha Fazer ', 'TVS Sport ', 'Honda Dream Yuga ', 'Yamaha FZ S ').
3. **Feature Derivation**: Create $\text{Car\_Age} = \text{Reference\_Year} - \text{Year}$ to provide a direct depreciation metric.
4. **Target Strategy**: Evaluate training models on both raw $\text{Selling\_Price}$ and $\log(1 + \text{Selling\_Price})$ to evaluate RMSE vs MAE trade-offs.
5. **High-Cardinality Strategy**: Rather than one-hot encoding 98 individual `Car_Name` values, extract brand names or rely on `Present_Price` as the proxy for vehicle segment tier.
6. **Leakage Prevention**: Train-test split before fitting any preprocessing scalers or encoders.
