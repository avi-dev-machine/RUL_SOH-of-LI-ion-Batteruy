"""
Data Loader for Battery Dataset
================================
Handles loading, cleaning, and basic preprocessing of the battery dataset.

Dataset Columns:
    battery_id  : Battery identifier (B5, B6, B7, ...)
    cycle       : Charge/discharge cycle number
    chI         : Charging current (A)
    chV         : Charging voltage (V)
    chT         : Charging temperature (°C)
    disI        : Discharging current (A)
    disV        : Discharging voltage (V)
    disT        : Discharging time (min)
    BCt         : Battery capacity (Ah)
    SOH         : State of Health (%)        ← TARGET
    RUL         : Remaining Useful Life (cycles) ← TARGET
"""

import os
import pandas as pd
import numpy as np

# Path to the dataset (one level up from this file's folder)
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Battery_dataset (1).csv"
)

FEATURE_COLS = ['cycle', 'chI', 'chV', 'chT', 'disI', 'disV', 'disT', 'BCt']
TARGET_COLS  = ['SOH', 'RUL']


def load_dataset(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the battery dataset from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            f"Please ensure 'Battery_dataset (1).csv' is in the parent directory."
        )
    df = pd.read_csv(path)
    print(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Batteries : {list(df['battery_id'].unique())}")
    print(f"   Columns   : {list(df.columns)}")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, handle missing values, fix dtypes."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"⚠️  Removed {before - after} duplicate rows.")

    missing = df.isnull().sum()
    if missing.any():
        print(f"⚠️  Missing values found:\n{missing[missing > 0]}")
        df = df.dropna()

    df['battery_id'] = df['battery_id'].astype(str)
    df['cycle']      = df['cycle'].astype(int)
    df['RUL']        = df['RUL'].astype(int)

    print(f"✅ Clean dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def get_features_targets(df: pd.DataFrame, target: str = 'SOH') -> tuple:
    """
    Extract feature matrix X and target vector y.

    Args:
        df     : Clean DataFrame
        target : 'SOH' or 'RUL'

    Returns:
        (X: DataFrame, y: Series)
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Available: {list(df.columns)}")
    X = df[FEATURE_COLS].copy()
    y = df[target].copy()
    return X, y


def get_battery_split(df: pd.DataFrame,
                       train_batteries: list = None,
                       test_batteries: list = None) -> tuple:
    """
    Split data by battery ID (for cross-battery generalization).
    Defaults to first 80% batteries for train, rest for test.
    """
    all_batteries = df['battery_id'].unique().tolist()

    if train_batteries is None and test_batteries is None:
        split_idx       = max(1, int(len(all_batteries) * 0.8))
        train_batteries = all_batteries[:split_idx]
        test_batteries  = all_batteries[split_idx:] if split_idx < len(all_batteries) else [all_batteries[-1]]

    train_df = df[df['battery_id'].isin(train_batteries)].copy()
    test_df  = df[df['battery_id'].isin(test_batteries)].copy()

    print(f"📊 Train batteries: {train_batteries} ({len(train_df)} samples)")
    print(f"📊 Test  batteries: {test_batteries}  ({len(test_df)} samples)")
    return train_df, test_df


def describe_dataset(df: pd.DataFrame) -> None:
    """Print a detailed dataset summary."""
    print("\n" + "=" * 65)
    print("  DATASET SUMMARY")
    print("=" * 65)
    print(f"  Shape      : {df.shape}")
    print(f"  Batteries  : {list(df['battery_id'].unique())}")
    print(f"  Cycle Range: {df['cycle'].min()} – {df['cycle'].max()}")
    print(f"  SOH Range  : {df['SOH'].min():.2f}% – {df['SOH'].max():.2f}%")
    print(f"  RUL Range  : {df['RUL'].min()} – {df['RUL'].max()} cycles")
    print("\n  Per-Battery Summary:")
    for bid in df['battery_id'].unique():
        bdf = df[df['battery_id'] == bid]
        print(f"    {bid:4s} | cycles: {len(bdf):4d} | "
              f"SOH: {bdf['SOH'].min():.1f}%–{bdf['SOH'].max():.1f}% | "
              f"RUL: {bdf['RUL'].min()}–{bdf['RUL'].max()}")
    print("\n  Descriptive Statistics:")
    print(df[FEATURE_COLS + TARGET_COLS].describe().round(4).to_string())
    print("=" * 65 + "\n")
