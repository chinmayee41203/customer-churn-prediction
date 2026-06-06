"""
Phase 4: Data Cleaning
- Drop irrelevant, leakage, and zero-variance columns
- Fix Total Charges dtype
- Handle missing values
- Remove duplicates
- Save cleaned data to data/processed/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


# ── Columns to drop (from EDA findings) ──────────────────────────────────────

# Zero variance — all same value
ZERO_VARIANCE = ['Country', 'State', 'Count']

# Identifiers / too granular for modeling
IDENTIFIERS = ['CustomerID', 'Lat Long', 'City', 'Zip Code', 'Latitude', 'Longitude']

# Data leakage — generated from churn label, not available at prediction time
LEAKAGE = ['Churn Score', 'CLTV']

# Redundant target — we keep 'Churn Value' (0/1)
REDUNDANT_TARGET = ['Churn Label']

# Only exists for churned customers — not useful as feature
HIGH_MISSING = ['Churn Reason']

DROP_COLS = ZERO_VARIANCE + IDENTIFIERS + LEAKAGE + REDUNDANT_TARGET + HIGH_MISSING


def load_data(filepath: str = None) -> pd.DataFrame:
    """Load raw Excel data."""
    path = filepath or (RAW_DATA_DIR / 'Telco_customer_churn.xlsx')
    df = pd.read_excel(path, engine='openpyxl')
    print(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop irrelevant, leakage, and redundant columns."""
    before = df.shape[1]
    df = df.drop(columns=DROP_COLS, errors='ignore')
    after = df.shape[1]
    print(f"✅ Dropped {before - after} columns → {after} remaining")
    print(f"   Dropped: {DROP_COLS}")
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Fix Total Charges — stored as object, should be float."""
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    nulls = df['Total Charges'].isnull().sum()
    print(f"✅ Fixed 'Total Charges' dtype → float64 ({nulls} NaN introduced)")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle remaining missing values.
    Total Charges NaN → customers with 0 tenure, fill with Monthly Charges.
    """
    before = df.isnull().sum().sum()

    # Total Charges NaN occurs when Tenure Months == 0 (new customers)
    mask = df['Total Charges'].isnull()
    df.loc[mask, 'Total Charges'] = df.loc[mask, 'Monthly Charges']
    print(f"✅ Filled {mask.sum()} 'Total Charges' NaN with 'Monthly Charges' (new customers)")

    after = df.isnull().sum().sum()
    print(f"✅ Missing values: {before} → {after}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    removed = before - after
    print(f"✅ Duplicates removed: {removed} ({before} → {after} rows)")
    return df


def validate_cleaned(df: pd.DataFrame) -> None:
    """Final validation checks."""
    print("\n===== VALIDATION =====")
    print(f"Shape          : {df.shape}")
    print(f"Missing values : {df.isnull().sum().sum()}")
    print(f"Duplicates     : {df.duplicated().sum()}")
    print(f"Target balance :\n{df['Churn Value'].value_counts()}")
    print(f"\nFinal columns  :\n{list(df.columns)}")
    print(f"\nDtypes:\n{df.dtypes}")


def save_cleaned(df: pd.DataFrame, filename: str = 'telco_churn_cleaned.csv') -> None:
    """Save cleaned data to data/processed/."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DATA_DIR / filename
    df.to_csv(out, index=False)
    print(f"\n✅ Cleaned data saved to: {out}")


def run_cleaning_pipeline(filepath: str = None) -> pd.DataFrame:
    """Run the full Phase 4 cleaning pipeline."""
    print("=" * 50)
    print("PHASE 4: DATA CLEANING")
    print("=" * 50)

    df = load_data(filepath)
    df = drop_columns(df)
    df = fix_dtypes(df)
    df = handle_missing(df)
    df = remove_duplicates(df)
    validate_cleaned(df)
    save_cleaned(df)

    print("\n✅ Phase 4 Complete!")
    return df


if __name__ == "__main__":
    df_clean = run_cleaning_pipeline()
