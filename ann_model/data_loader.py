"""
Data Loader for ANN Battery Model
===================================
Loads the battery dataset, optionally engineers derived features,
and provides train / validation / test splits.

Dataset columns:
    battery_id  — Battery identifier (B5, B6, ...)
    cycle       — Charge/discharge cycle index
    chI         — Charging current (A)
    chV         — Charging voltage (V)
    chT         — Charging temperature (°C)
    disI        — Discharging current (A)
    disV        — Discharging voltage (V)
    disT        — Discharging time (min)
    BCt         — Battery capacity (Ah)
    SOH         — State of Health (%)         ← target
    RUL         — Remaining Useful Life (cyc) ← target
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Battery_dataset (1).csv"
)

BASE_FEATURES    = ['cycle', 'chI', 'chV', 'chT', 'disI', 'disV', 'disT', 'BCt']
DERIVED_FEATURES = [
    'charge_energy', 'discharge_energy', 'energy_efficiency',
    'temp_diff', 'current_ratio', 'voltage_drop',
    'capacity_fade_rate', 'cycle_normalized',
]
TARGET_COLS = ['SOH', 'RUL']


# ── Raw loading ────────────────────────────────────────────────────────────────

def load_dataset(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and minimally clean the CSV dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Ensure 'Battery_dataset (1).csv' is in the parent folder."
        )
    df = pd.read_csv(path)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df['battery_id'] = df['battery_id'].astype(str)
    df['cycle']      = df['cycle'].astype(int)
    df['RUL']        = df['RUL'].astype(int)

    print(f"✅ Loaded {len(df)} samples | "
          f"Batteries: {sorted(df['battery_id'].unique())}")
    return df


# ── Feature engineering ────────────────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute physics-motivated derived features."""
    df = df.copy().sort_values(['battery_id', 'cycle']).reset_index(drop=True)

    df['charge_energy']    = df['chI'] * df['chV']
    df['discharge_energy'] = df['disI'] * df['disV']
    df['energy_efficiency'] = np.where(
        df['charge_energy'] > 0,
        df['discharge_energy'] / df['charge_energy'],
        np.nan
    )
    df['temp_diff']     = df['chT'] - df['disT']
    df['current_ratio'] = np.where(df['chI'] > 0, df['disI'] / df['chI'], np.nan)
    df['voltage_drop']  = df['chV'] - df['disV']

    # Capacity fade rate per battery
    df['capacity_fade_rate'] = np.nan
    for bid in df['battery_id'].unique():
        mask = df['battery_id'] == bid
        bdf  = df.loc[mask].sort_values('cycle')
        dr   = bdf['BCt'].diff() / bdf['cycle'].diff().replace(0, np.nan)
        df.loc[bdf.index, 'capacity_fade_rate'] = dr.values

    # Normalised cycle [0, 1] per battery
    df['cycle_normalized'] = np.nan
    for bid in df['battery_id'].unique():
        mask = df['battery_id'] == bid
        cyc  = df.loc[mask, 'cycle']
        span = cyc.max() - cyc.min() or 1
        df.loc[mask, 'cycle_normalized'] = (cyc - cyc.min()) / span

    # Impute any NaNs introduced
    for col in DERIVED_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def get_feature_list(use_derived: bool = True) -> list:
    """Return the ordered list of feature column names."""
    return BASE_FEATURES + DERIVED_FEATURES if use_derived else BASE_FEATURES


# ── Data splitting ─────────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame,
                  use_derived: bool = True,
                  test_size:   float = 0.20,
                  val_size:    float = 0.10,
                  random_state: int  = 42) -> dict:
    """
    Prepare numpy arrays for ANN training.

    Returns a dict with:
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_names, target_names
    """
    if use_derived:
        df = add_derived_features(df)

    features = get_feature_list(use_derived)
    features = [f for f in features if f in df.columns]   # safety filter

    X = df[features].values.astype(np.float32)
    y = df[TARGET_COLS].values.astype(np.float32)          # shape (N, 2)

    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size),
        random_state=random_state, shuffle=True
    )

    # Second split: val vs test from the temp portion
    val_frac = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - val_frac),
        random_state=random_state
    )

    print(f"📊 Split → Train: {len(X_train)}  "
          f"Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"   Features ({len(features)}): {features}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val':   X_val,   'y_val':   y_val,
        'X_test':  X_test,  'y_test':  y_test,
        'feature_names': features,
        'target_names' : TARGET_COLS,
    }
