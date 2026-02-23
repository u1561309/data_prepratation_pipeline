# Home Credit Default Risk – Data Preparation Pipeline & Modeling

This repository contains a reproducible data preparation pipeline and an advanced predictive machine learning sequence for the Home Credit Default Risk dataset. The ultimate goal is to process messy financial data accurately and construct models capable of classifying which clients are likely to default on their loan payments.

## Table of Contents
- [Introduction](#introduction)
- [Data Preparation](#data-preparation)
- [Modeling Process](#modeling-process)
- [Model Performance](#model-performance)
- [Results](#results)

## Introduction

The data pipeline and modeling sequence within this project are built sequentially to handle complex, heavily imbalanced financial data efficiently while avoiding critical evaluation failures (e.g. data leakages or uncalibrated cross-validations). 

The raw data (`application_train.csv`, `bureau.csv`, etc.) is unified and transformed dynamically inside `main.py` functions before being ingested into `Modeling.ipynb` for empirical algorithm comparisons and parameter tuning. 

---

## Data Preparation
The data preprocessing operations defined in `main.py` ensure safe, reproducible extraction of demographic, financial, and behavioral patterns.

### 1. Data Cleaning
- Transforms problematic artifacts automatically (e.g., placeholder `DAYS_EMPLOYED = 365243` replaced with NaNs).
- Casts negative day variables into meaningful operational units (Age & Employment Duration in Years).

### 2. Feature Engineering
- Derived important Financial Ratios: Credit-to-Income, Annuity-to-Income, and Loan-to-Value variants.
- Computed Aggregates corresponding to External Credit Source fields.
- Binning indicators for demographic and employment stability intervals.

### 3. Supplementary Data Aggregation
The pipeline groups, merges, and summarizes the following large supplementary tables against the principal client ID (`SK_ID_CURR`):
- `bureau.csv`
- `previous_application.csv`
- `installments_payments.csv`
- `POS_CASH_balance.csv` 
- `credit_card_balance.csv` 

This generates rich, highly expressive relational metrics such as *bureau debt ratio*, *previous approval rates*, and *late payment frequencies*.

### 4. Train/Test Alignment
To prevent data leakage, imputation medians are established *exclusively* on the training data and then blindly applied identically to the test distribution. 

---

## Modeling Process

The end-to-end learning deployment is documented and executable via `Modeling.ipynb`.

### Class Imbalance Strategies
Given the severe class imbalance (~8% default rate), two synthetic data strategies were evaluated for their impact on learning the minority class:
1. **Upsampling**: Replicated instances via `sklearn.utils.resample`.
2. **SMOTE**: Generative synthetic up-sampling via `imblearn`.

Ultimately, scaling the tree model weights contextually via inverse class distribution ratios (`scale_pos_weight`) proved safer and more reliable than artificially transforming the input dimensions.

### Hyperparameter Tuning
Robust tuning was deployed via structured searches (`RandomizedSearchCV`). To circumvent vast compilation times, hyperparameter grids were iterated over a stratified subset of **5,000 observations** mapping back through **3-fold cross-validation** pipelines.

---

## Model Performance

Initial benchmarks were compiled on base data features natively mapped onto ROC implementations. Area Under Curve (AUC) scores were observed across architectures:
- **Dummy Classifier**: 0.500
- **Logistic Regression**: ~0.615
- **Random Forest**: ~0.747
- **XGBoost**: ~0.766

XGBoost proved natively resilient and was selected as the foundational classifier.

Upon finalizing the architecture, hyperparameter subsets were tuned. Incorporating the final supplementary feature clusters (`POS_CASH_balance` & `credit_card_balance`) empirically pushed the grid-search subset generalization bounds upwards from **~0.735 AUC to ~0.739 AUC**. 

---

## Results

The final architecture selected is an `XGBClassifier` parameterized globally by localized grid outputs:
`{'subsample': 0.8, 'scale_pos_weight': 11.37, 'n_estimators': 100, 'min_child_weight': 1, 'max_depth': 3, 'learning_rate': 0.05, 'gamma': 0.3, 'colsample_bytree': 1.0}`

Executing this parameterized model broadly over the fully unified demographic and supplemental datasets achieved a robust, fully-generalized CV **AUC of ~0.754**. 

This trained framework was then tasked onto the independent `application_test.csv` dataset, and raw submission probabilities were effectively pushed and formatted inside `/data_preparation/submission.csv` for Kaggle evaluation scoring.