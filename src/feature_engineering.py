"""
Phase 5: Feature Engineering
- Encode binary categories (Yes/No → 0/1)
- Encode multi-class categories (one-hot or ordinal)
- Create new features from domain knowledge
- Scale numerical features
- Save final feature matrix to data/processed/
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE


# ── Encoding Maps ─────────────────────────────────────────────────────────────

# Binary Yes/No columns → 0/1
BINARY_YES_NO = [
    'Partner', 'Dependents', 'Phone Service',
    'Paperless Billing', 'Senior Citizen'
]

# Gender
GENDER_MAP = {'Male': 1, 'Female': 0}

# Contract → ordinal (longer contract = higher value)
CONTRACT_MAP = {
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2
}

# 3-value columns with "No internet/phone service" → treat as 0
THREE_WAY_COLS = [
    'Multiple Lines', 'Online Security', 'Online Backup',
    'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies'
]

# One-hot encode these
OHE_COLS = ['Internet Service', 'Payment Method']


def load_cleaned(filepath: str = None) -> pd.DataFrame:
    """Load Phase 4 cleaned data."""
    path = filepath or (PROCESSED_DATA_DIR / 'telco_churn_cleaned.csv')
    df = pd.read_csv(path)
    print(f"✅ Loaded cleaned data: {df.shape}")
    return df


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Yes/No columns to 1/0."""
    for col in BINARY_YES_NO:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    df['Gender'] = df['Gender'].map(GENDER_MAP)
    print(f"✅ Binary encoded: {BINARY_YES_NO + ['Gender']}")
    return df


def encode_three_way(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode 3-value service columns:
    'Yes' → 1, 'No' → 0, 'No internet/phone service' → 0
    """
    for col in THREE_WAY_COLS:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 
                                   'No internet service': 0,
                                   'No phone service': 0})
    print(f"✅ Three-way encoded: {THREE_WAY_COLS}")
    return df


def encode_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal encode Contract (month-to-month=0, one year=1, two year=2)."""
    df['Contract'] = df['Contract'].map(CONTRACT_MAP)
    print(f"✅ Ordinal encoded: Contract → {CONTRACT_MAP}")
    return df


def encode_ohe(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Internet Service and Payment Method."""
    df = pd.get_dummies(df, columns=OHE_COLS, drop_first=False, dtype=int)
    print(f"✅ One-hot encoded: {OHE_COLS}")
    return df


def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features from domain knowledge.
    """
    # Charges per month of tenure (efficiency metric)
    df['Charges Per Month'] = (
        df['Total Charges'] / (df['Tenure Months'] + 1)  # +1 avoids div by zero
    ).round(2)

    # Total number of services subscribed
    service_cols = [
        'Phone Service', 'Multiple Lines', 'Online Security',
        'Online Backup', 'Device Protection', 'Tech Support',
        'Streaming TV', 'Streaming Movies'
    ]
    # These are already encoded 0/1 at this point
    df['Num Services'] = df[service_cols].sum(axis=1)

    # Tenure group (binned)
    df['Tenure Group'] = pd.cut(
        df['Tenure Months'],
        bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3],  # numeric labels for modeling
        include_lowest=True
    ).astype(int)

    # High value customer flag (monthly charges > 75th percentile)
    threshold = df['Monthly Charges'].quantile(0.75)
    df['High Value Customer'] = (df['Monthly Charges'] > threshold).astype(int)

    # Auto payment flag (bank transfer or credit card)
    auto_pay_cols = [c for c in df.columns if 'Bank transfer' in c or 'Credit card' in c]
    if auto_pay_cols:
        df['Auto Payment'] = df[auto_pay_cols].max(axis=1)

    print(f"✅ New features created: Charges Per Month, Num Services, Tenure Group, High Value Customer, Auto Payment")
    return df


def scale_features(df: pd.DataFrame, target_col: str = 'Churn Value') -> tuple:
    """
    Standard scale numerical features.
    Returns scaled df and fitted scaler.
    """
    num_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'Charges Per Month']
    num_cols = [c for c in num_cols if c in df.columns]

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Save scaler
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')

    print(f"✅ Scaled features: {num_cols}")
    print(f"✅ Scaler saved to: models/scaler.pkl")
    return df, scaler


def save_features(df: pd.DataFrame, filename: str = 'telco_churn_features.csv') -> None:
    """Save final feature matrix."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DATA_DIR / filename
    df.to_csv(out, index=False)
    print(f"✅ Feature matrix saved to: {out}")
    print(f"   Final shape: {df.shape}")


def run_feature_pipeline(filepath: str = None) -> pd.DataFrame:
    """Run the full Phase 5 feature engineering pipeline."""
    print("=" * 50)
    print("PHASE 5: FEATURE ENGINEERING")
    print("=" * 50)

    df = load_cleaned(filepath)
    df = encode_binary(df)
    df = encode_three_way(df)
    df = encode_contract(df)
    df = encode_ohe(df)
    df = create_new_features(df)
    df, scaler = scale_features(df)
    save_features(df)

    print("\n✅ Phase 5 Complete!")
    print(f"Features: {[c for c in df.columns if c != 'Churn Value']}")
    return df


if __name__ == "__main__":
    df_features = run_feature_pipeline()
