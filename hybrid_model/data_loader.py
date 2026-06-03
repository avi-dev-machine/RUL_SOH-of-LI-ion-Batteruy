"""
Data Loader for Hybrid Physics-Informed ANN
=============================================
Loads the battery dataset and provides train/val/test splits
for the hybrid pipeline.

Dataset: Battery_dataset (1).csv
    battery_id, cycle, chI, chV, chT, disI, disV, disT, BCt, SOH, RUL
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Battery_dataset (1).csv"
)

# The 14 engineered features referenced in the Hybrid System Architecture (Fig 12.1)
FEATURE_14 = [
    # 8 raw measurements
    'cycle', 'chI', 'chV', 'chT', 'disI', 'disV', 'disT', 'BCt',
    # 6 derived
    'charge_energy', 'discharge_energy', 'energy_efficiency',
    'temp_diff', 'current_ratio', 'voltage_drop',
]

# Physics features added by the Physics Engine
PHYSICS_FEATURES = [
    'SOH_phys', 'RUL_phys_norm', 'PDI',
    'thermal_stress', 'current_stress',
]

TARGET_COLS = ['SOH', 'RUL']


def load_dataset(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and minimally clean the battery CSV dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df['battery_id'] = df['battery_id'].astype(str)
    df['cycle']      = df['cycle'].astype(int)
    df['RUL']        = df['RUL'].astype(int)

    print(f"[DataLoader] Loaded {len(df)} samples | "
          f"Batteries: {sorted(df['battery_id'].unique())}")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 6 derived features to reach 14 total (Fig 12.1)."""
    df = df.copy().sort_values(['battery_id', 'cycle']).reset_index(drop=True)

    df['charge_energy']    = df['chI'] * df['chV']
    df['discharge_energy'] = df['disI'] * df['disV']
    df['energy_efficiency'] = np.where(
        df['charge_energy'] > 0,
        df['discharge_energy'] / df['charge_energy'], np.nan
    )
    df['temp_diff']     = df['chT'] - df['disT']
    df['current_ratio'] = np.where(df['chI'] > 0, df['disI'] / df['chI'], np.nan)
    df['voltage_drop']  = df['chV'] - df['disV']

    for col in ['energy_efficiency', 'current_ratio']:
        df[col] = df[col].fillna(df[col].median())

    return df


def split_data(df:           pd.DataFrame,
               feature_cols: list,
               test_size:    float = 0.20,
               val_size:     float = 0.10,
               random_state: int   = 42) -> dict:
    """
    Split into train / val / test sets.

    Returns dict with:
        X_train, y_train, X_val, y_val, X_test, y_test
        y_phys_train, y_phys_val, y_phys_test  (physics SOH/RUL for loss)
        meta_test  (DataFrame with battery_id + cycle for the test rows)
    """
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COLS].values.astype(np.float32)

    # Physics targets (SOH_phys and RUL_phys_norm) used in physics loss
    phys_cols = [c for c in PHYSICS_FEATURES if c in df.columns]
    y_phys    = df[phys_cols].values.astype(np.float32) if phys_cols else np.zeros((len(df), 2), dtype=np.float32)

    idx = np.arange(len(df))
    idx_train, idx_temp = train_test_split(idx, test_size=(test_size + val_size),
                                            random_state=random_state, shuffle=True)
    val_frac = val_size / (test_size + val_size)
    idx_val, idx_test   = train_test_split(idx_temp, test_size=(1 - val_frac),
                                            random_state=random_state)

    meta_test = df.iloc[idx_test][['battery_id', 'cycle']].reset_index(drop=True)

    print(f"[DataLoader] Train: {len(idx_train)}  Val: {len(idx_val)}  Test: {len(idx_test)}")

    return {
        'X_train'      : X[idx_train],  'y_train'      : y[idx_train],
        'X_val'        : X[idx_val],    'y_val'        : y[idx_val],
        'X_test'       : X[idx_test],   'y_test'       : y[idx_test],
        'y_phys_train' : y_phys[idx_train],
        'y_phys_val'   : y_phys[idx_val],
        'y_phys_test'  : y_phys[idx_test],
        'meta_test'    : meta_test,
        'feature_cols' : feature_cols,
    }
