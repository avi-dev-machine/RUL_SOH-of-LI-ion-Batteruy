"""
Feature Engineering for Battery Analytical Model
==================================================
Creates physically-motivated derived features from raw battery measurements.

Derived Features:
    charge_energy       : chI × chV  (proxy for energy stored)
    discharge_energy    : disI × disV (proxy for energy released)
    energy_efficiency   : discharge_energy / charge_energy (Coulombic efficiency proxy)
    temp_diff           : chT - disT  (thermal stress indicator)
    current_ratio       : disI / chI  (C-rate asymmetry)
    voltage_drop        : chV - disV  (internal resistance indicator)
    capacity_fade_rate  : dBCt/d(cycle) per battery (degradation velocity)
    cycle_normalized    : cycle scaled [0,1] per battery
"""

import numpy as np
import pandas as pd


BASE_FEATURES = ['cycle', 'chI', 'chV', 'chT', 'disI', 'disV', 'disT', 'BCt']

DERIVED_FEATURES = [
    'charge_energy', 'discharge_energy', 'energy_efficiency',
    'temp_diff', 'current_ratio', 'voltage_drop',
    'capacity_fade_rate', 'cycle_normalized'
]


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all derived features from raw measurements.

    Args:
        df: Clean DataFrame with base features

    Returns:
        DataFrame with additional derived columns appended
    """
    df = df.copy()

    # ── Energy proxies ─────────────────────────────────────────────
    df['charge_energy']    = df['chI'] * df['chV']
    df['discharge_energy'] = df['disI'] * df['disV']

    # Guard against division by zero
    df['energy_efficiency'] = np.where(
        df['charge_energy'] > 0,
        df['discharge_energy'] / df['charge_energy'],
        np.nan
    )

    # ── Electrical / thermal ratios ────────────────────────────────
    df['temp_diff']     = df['chT'] - df['disT']
    df['current_ratio'] = np.where(df['chI'] > 0, df['disI'] / df['chI'], np.nan)
    df['voltage_drop']  = df['chV'] - df['disV']

    # ── Capacity fade rate (dBCt / d_cycle) per battery ───────────
    df = df.sort_values(['battery_id', 'cycle']).reset_index(drop=True)
    df['capacity_fade_rate'] = np.nan
    for bid in df['battery_id'].unique():
        mask = df['battery_id'] == bid
        bdf  = df.loc[mask].sort_values('cycle')
        delta_bct   = bdf['BCt'].diff()
        delta_cycle = bdf['cycle'].diff().replace(0, np.nan)
        fade = delta_bct / delta_cycle
        df.loc[bdf.index, 'capacity_fade_rate'] = fade.values

    # ── Normalized cycle per battery [0, 1] ───────────────────────
    df['cycle_normalized'] = np.nan
    for bid in df['battery_id'].unique():
        mask = df['battery_id'] == bid
        cyc  = df.loc[mask, 'cycle']
        c_min, c_max = cyc.min(), cyc.max()
        denom = c_max - c_min if c_max != c_min else 1.0
        df.loc[mask, 'cycle_normalized'] = (cyc - c_min) / denom

    # ── Fill NaN from first-cycle differences ─────────────────────
    df['capacity_fade_rate'] = df['capacity_fade_rate'].fillna(0.0)
    df['energy_efficiency']  = df['energy_efficiency'].fillna(df['energy_efficiency'].median())
    df['current_ratio']      = df['current_ratio'].fillna(df['current_ratio'].median())

    print(f"✅ Feature engineering complete: {df.shape[1]} total columns")
    return df


def get_all_features(with_derived: bool = True) -> list:
    """Return the list of feature column names to use for modelling."""
    if with_derived:
        return BASE_FEATURES + DERIVED_FEATURES
    return BASE_FEATURES


def print_feature_stats(df: pd.DataFrame) -> None:
    """Print descriptive statistics for all engineered features."""
    all_feats = get_all_features(with_derived=True)
    available = [f for f in all_feats if f in df.columns]
    print("\n  Engineered Feature Statistics:")
    print(df[available].describe().round(4).to_string())
