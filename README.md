# Home Credit Default Risk – Data Preparation Pipeline

This repository contains a reusable data preparation pipeline for the
Home Credit Default Risk dataset. The goal of this project is to clean,
transform, and engineer features in a train/test-safe manner based on
exploratory data analysis (EDA).

This project focuses on **data engineering and preprocessing**, not
model training or prediction.

---

## Project Overview

The data preparation pipeline is implemented in `main.py` and is designed
to be reusable for both training and test datasets. All transformations
are implemented as functions and can be composed into a preprocessing
workflow.

Key objectives:
- Fix known data quality issues identified during EDA
- Engineer meaningful demographic, financial, and behavioral features
- Aggregate supplementary datasets to the applicant level (`SK_ID_CURR`)
- Ensure train/test consistency and prevent data leakage

---

## Data Preparation (`main.py`)

The `main.py` script performs the following tasks:

### 1. Data Cleaning
- Fixes the placeholder value `DAYS_EMPLOYED = 365243` by converting it to missing
- Converts day-based variables into meaningful units (age and employment duration in years)

### 2. Feature Engineering
- Demographic features (age, employment duration)
- Financial ratios such as:
  - Credit-to-income ratio
  - Annuity-to-income ratio
  - Loan-to-value ratio
- Aggregate EXT_SOURCE features (mean, min, max)
- Missing value indicators
- Binned and interaction features (e.g., age bins, employment stability flags)

### 3. Supplementary Data Aggregation
The following datasets are aggregated to the applicant level (`SK_ID_CURR`):
- `bureau.csv`
- `previous_application.csv`
- `installments_payments.csv`

Aggregations include:
- Counts of prior credits and applications
- Approval and refusal history
- Debt and overdue ratios
- Late payment rates

### 4. Train/Test Consistency
- Imputation statistics (medians) are computed using **training data only**
- The same statistics are applied to test data
- Final train and test datasets have identical feature columns (except `TARGET`)

---

## Usage

The functions in `main.py` are intended to be imported and used within
a notebook or modeling pipeline.

Example usage:

```python
import pandas as pd

from main import (
    fix_days_employed,
    convert_days_to_years,
    add_ext_source_features,
    add_financial_ratios,
    add_binned_and_interaction_features,
    aggregate_bureau,
    aggregate_previous_applications,
    aggregate_installments,
    join_aggregates,
    align_train_test
)
```
Load application data
```
train_df = pd.read_csv("data/application_train.csv")
test_df  = pd.read_csv("data/application_test.csv")
```
Load supplementary datasets
```
bureau_df = pd.read_csv("data/bureau.csv")
prev_df   = pd.read_csv("data/previous_application.csv")
inst_df   = pd.read_csv("data/installments_payments.csv")
```

Clean and engineer features

```
for df in [train_df, test_df]:
    df = fix_days_employed(df)
    df = convert_days_to_years(df)
    df = add_ext_source_features(df)
    df = add_financial_ratios(df)
    df = add_binned_and_interaction_features(df)
```

Aggregate auxiliary tables

```
bureau_agg = aggregate_bureau(bureau_df)
prev_agg   = aggregate_previous_applications(prev_df)
inst_agg   = aggregate_installments(inst_df)
```

Join aggregated features
```
train_df = join_aggregates(train_df, bureau_agg, prev_agg, inst_agg)
test_df  = join_aggregates(test_df, bureau_agg, prev_agg, inst_agg)
```
Align train and test columns

```
train_df, test_df = align_train_test(train_df, test_df)
```