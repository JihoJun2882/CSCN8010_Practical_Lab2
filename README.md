# Practical Lab 2 — Diabetes Progression (scikit-learn)

Predict **one-year diabetes progression** from clinical features using the scikit-learn diabetes dataset. This project walks through **EDA → data prep → modeling (uni/multivariate) → validation/test evaluation → conclusions** with transparent, reproducible code.

## Dataset

- **Source:** `sklearn.datasets.load_diabetes`
- **Target (`y`):** disease progression score **1 year after baseline** (continuous).
- **Features (`X`, standardized):** `age, sex, bmi, bp, s1, s2, s3, s4, s5, s6`
  - `bmi`: body mass index
  - `bp`: average blood pressure
  - `s1`: total serum cholesterol (tc)
  - `s2`: LDL (low-density lipoproteins)
  - `s3`: HDL (high-density lipoproteins)
  - `s4`: tc/HDL ratio
  - `s5`: log(triglycerides) (approx.)
  - `s6`: blood glucose level
- **Interpretation note:** `X` is standardized, `y` is not. Linear coefficients are interpreted as **change in `y` per 1 standard deviation change** in the feature.

## Environment & Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook Practical_Lab2.ipynb
```

## Workflow

### Part 1 — Data & EDA
1. **Get the data:** load with `load_diabetes()` and build a `pandas.DataFrame`.
2. **Frame the problem:** regression; predict 1-year progression. **Metrics:** R² (higher is better), MAE/MAPE (lower is better).
3. **EDA:** summary statistics, histograms, scatter plots vs target, correlation matrix. Record concise insights (e.g., BMI has strong positive relationship with the target; lipid variables are intercorrelated).
4. **Cleaning:** check dtypes, missing values, constant/ID-like columns. **Feature drop used in this notebook:** drop **`sex`** (by EDA choice). If you prefer, keep all features.
5. **Split:** **train 75% / validation 10% / test 15%** with fixed `random_state` for reproducibility.

### Part 2 — Univariate Polynomial on BMI (degrees 0–5)
- Fit 6 models: polynomial degree **0–5** using a pipeline  
  `PolynomialFeatures(include_bias=True) → LinearRegression(fit_intercept=False)`  
  (the constant term comes from the bias column to avoid duplicate intercepts).
- Compute **train/val** metrics (R², MAE, MAPE) and select the **best degree by validation**.
- Evaluate the selected model on the **test** set.
- **Plot:** train/val/test scatter and the fitted curve for the chosen degree.
- **Equation:** print the fitted polynomial using `get_feature_names_out()`; format coefficients to **2 decimals**.
- **Single-point prediction:** `model.predict([[bmi_z]])` for a standardized BMI value (e.g., `0.0`, `0.5`, `-0.5`).

### Part 3 — All Features (or drop based on EDA)
Repeat the Part 2 selection workflow with full features:
1. **Polynomial regression ×2** (degrees > 1, e.g., 2 and 3).
2. **Decision Tree Regressor ×2** (e.g., `max_depth=3` and `max_depth=4`).
3. **kNN Regressor ×2** (e.g., `k=3` and `k=10`).
4. **Logistic Regression ×2 (classification)**  
   - Convert continuous target to **binary labels** by the **train-set median** (use the same threshold for val/test to avoid leakage).  
   - **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC.  
   - **Selection:** maximize **F1_val** (or ROC_AUC_val) on validation.

## Model Selection Policy

- **Always choose using validation metrics** (never peek at the test set):  
  - **Regression families (poly/tree/kNN):** pick the highest **R²_val** (break ties with **MAE_val**/**MAPE_val**).  
  - **Classification (logistic):** pick the highest **F1_val** (or **ROC_AUC_val**).
- After selection, report the chosen model’s **validation and test** metrics side by side for transparency.

## Reproducibility Notes

- Use a fixed `random_state` for splitting and any stochastic models.  
- Because splits are random, **numbers will vary** across runs; evaluate comparatively (VAL first, TEST second).  
- `X` is already standardized in `load_diabetes`; extra scaling is optional for tree models but generally helpful for kNN/linear.

## Example Results (will vary with your split)

- **Feature drop:** `sex` only.  
- **Validation winners (per family):** `poly_deg2`, `tree_depth3`, `knn_k10`, `logreg_L2_C0.1_balanced`.  
- **Overall best by validation (regression):** `poly_deg2`.  
- Although `poly_deg2` and `kNN` (e.g., `knn_k15`) can show similar R², **`poly_deg2`** was preferred because **validation and test scores are closely aligned**, indicating better generalization/stability.

## How to Reproduce

1. Run the **EDA** cells to inspect distributions and correlations.  
2. Execute **Part 2** to compare BMI polynomials (degrees 0–5) and plot the best fit.  
3. Execute **Part 3** to train the two candidates per family (poly/tree/kNN/logistic).  
4. **Select by validation** (per family and optionally overall across regression families).  
5. **Evaluate on test** the selected model(s) and summarize.

## Reporting Templates

**Regression (validation):**
```text
Model       R2_val   MAE_val   MAPE_val
----------  ------   -------   --------
poly_deg2    0.39     42.0      41.0%
tree_depth3  0.20     48.1      47.7%
knn_k10      0.39     42.6      39.1%
```

**Regression (test):**
```text
Model       R2_test  MAE_test  MAPE_test
----------  -------  --------  ---------
poly_deg2    0.39     47.5      38.8%
tree_depth3  0.44     46.1      40.2%
knn_k10      0.50     41.9      35.1%
```

**Classification (validation/test):**
```text
Model                   F1_val  ROC_AUC_val  ACC_val | F1_test  ROC_AUC_test  ACC_test
---------------------   ------  -----------  ------- | -------  ------------  --------
logreg_L2_C0.1_balanced  0.78      0.89       0.78   |  0.78        0.89        0.78
```
> Replace with your actual outputs from the notebook.

## Conclusions & Limitations

- **BMI-only** (Part 2) captures a nonlinear positive trend but has **limited explanatory power**; biological outcomes are multi-factor.  
- **High-BMI** ranges can show **larger errors** (heteroscedasticity). Including additional features (e.g., `bp`, `s5`) improves fit.  
- **Polynomials**: higher degrees risk **overfitting**; rely on validation and consider **Ridge/Lasso** if allowed.  
- **Trees/kNN**: capture nonlinear/local patterns; tune `max_depth` and `k`.  
- **Logistic** (median split): useful for thresholded risk but loses continuous detail.

## Next Steps

- Add **K-Fold cross-validation** to stabilize validation metrics.  
- Try **regularization** (Ridge/Lasso) or **feature selection/PCA** (lipid multicollinearity).  
- Compare **tree ensembles** (RandomForest/GBDT) and **linear models with interactions**.

## Project Structure

```
Practical_Lab2.ipynb   # main notebook (EDA, modeling, evaluation, plots)
README.md              # this file
```

## Attribution

- Dataset from **scikit-learn**: Diabetes dataset.  
- Built with **NumPy, pandas, scikit-learn, Matplotlib/Seaborn**.
