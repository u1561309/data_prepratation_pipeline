"""
Reusable data preparation pipeline for the Home Credit Default Risk project.

This script defines functions to:
- Clean known data issues identified during EDA
- Engineer demographic, financial, and behavioral features
- Aggregate supplementary datasets to the applicant level (SK_ID_CURR)
- Handle missing values in a train/test consistent manner

IMPORTANT:
- This script does NOT read or write any data files.
- It is designed to be imported and used in notebooks or modeling pipelines.
- All statistics (medians, bins) must be computed on TRAIN data only
  and reused for TEST data.
"""

import numpy as np
import pandas as pd

# =====================================================
# 1. DATA CLEANING
# =====================================================

def fix_days_employed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace placeholder value 365243 in DAYS_EMPLOYED with NaN.
    """
    df = df.copy()
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    return df


def convert_days_to_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert negative day-based variables to positive values in years.
    """
    df = df.copy()
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365
    return df


# =====================================================
# 2. MISSING VALUE HANDLING
# =====================================================

def add_missing_indicators(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Add binary indicators for missing values.
    Missingness itself is often predictive.
    """
    df = df.copy()
    for col in columns:
        df[f"{col}_MISSING"] = df[col].isna().astype(int)
    return df


def fit_numeric_imputers(train_df: pd.DataFrame, columns: list) -> dict:
    """
    Compute median values from training data only.
    """
    return train_df[columns].median().to_dict()


def apply_numeric_imputers(df: pd.DataFrame, imputers: dict) -> pd.DataFrame:
    """
    Apply precomputed medians to fill missing values.
    """
    df = df.copy()
    for col, value in imputers.items():
        df[col] = df[col].fillna(value)
    return df


# =====================================================
# 3. FEATURE ENGINEERING
# =====================================================

def add_ext_source_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated EXT_SOURCE features.
    """
    df = df.copy()
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

    df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
    df["EXT_SOURCE_MAX"] = df[ext_cols].max(axis=1)
    df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)

    return df


def add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common and useful financial ratios for credit risk modeling.
    """
    df = df.copy()

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["CREDIT_PER_PERSON"] = df["AMT_CREDIT"] / df["CNT_FAM_MEMBERS"]

    return df


def add_binned_and_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binned demographic variables and interaction-style features.
    """
    df = df.copy()

    # Age bins commonly used in credit scoring
    df["AGE_BIN"] = pd.cut(
        df["AGE_YEARS"],
        bins=[18, 25, 35, 45, 55, 65, 100],
        labels=["18-25", "25-35", "35-45", "45-55", "55-65", "65+"]
    )

    # Employment stability indicators
    df["LONG_EMPLOYED"] = (df["EMPLOYED_YEARS"] > 5).astype(int)
    df["VERY_LONG_EMPLOYED"] = (df["EMPLOYED_YEARS"] > 10).astype(int)

    return df


# =====================================================
# 4. SUPPLEMENTARY DATA AGGREGATION
# =====================================================

def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate bureau data to applicant level (SK_ID_CURR).
    """
    agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        BUREAU_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
    )

    agg["BUREAU_DEBT_RATIO"] = agg["BUREAU_DEBT_SUM"] / agg["BUREAU_CREDIT_SUM"]
    return agg.reset_index()


def aggregate_previous_applications(prev: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate previous application history.
    """
    agg = prev.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
        PREV_REFUSED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
    )

    agg["PREV_APPROVAL_RATE"] = agg["PREV_APPROVED_COUNT"] / agg["PREV_APP_COUNT"]
    return agg.reset_index()


def aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate installment payment behavior.
    """
    inst = inst.copy()
    inst["LATE_PAYMENT"] = (
        inst["DAYS_ENTRY_PAYMENT"] > inst["DAYS_INSTALMENT"]
    ).astype(int)

    agg = inst.groupby("SK_ID_CURR").agg(
        INSTALLMENT_COUNT=("SK_ID_PREV", "count"),
        LATE_PAYMENT_RATE=("LATE_PAYMENT", "mean"),
    )

    return agg.reset_index()


# =====================================================
# 5. FINAL DATASET ASSEMBLY
# =====================================================

def join_aggregates(
    app: pd.DataFrame,
    bureau_agg: pd.DataFrame,
    prev_agg: pd.DataFrame,
    inst_agg: pd.DataFrame
) -> pd.DataFrame:
    """
    Join all aggregated supplementary datasets to application data.
    """
    app = app.merge(bureau_agg, on="SK_ID_CURR", how="left")
    app = app.merge(prev_agg, on="SK_ID_CURR", how="left")
    app = app.merge(inst_agg, on="SK_ID_CURR", how="left")
    return app


def align_train_test(train: pd.DataFrame, test: pd.DataFrame):
    """
    Ensure train and test have identical columns (except TARGET).
    """
    train_cols = set(train.columns)
    test_cols = set(test.columns)

    for col in train_cols - test_cols:
        if col != "TARGET":
            test[col] = 0

    for col in test_cols - train_cols:
        test.drop(columns=col, inplace=True)

    return train, test