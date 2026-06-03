"""
Feature Fusion Layer
=====================
Implements the Feature Fusion Layer block from the Hybrid System Architecture (Fig 12.1).

Combines:
    - 14 engineered features (raw measurements + derived)
    - 5 physics features (SOH_phys, RUL_phys_norm, PDI, thermal_stress, current_stress)

Total fused feature vector: 19 dimensions per sample.

The fusion is a simple concatenation — the Physics-Informed ANN then learns
optimal weighting between data-driven and physics-based signals.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 14 engineered features (Figure 12.1: "Feature Engineering (14 features)")
FEATURES_14 = [
    # Raw measurements (8)
    'cycle', 'chI', 'chV', 'chT', 'disI', 'disV', 'disT', 'BCt',
    # Derived (6)
    'charge_energy', 'discharge_energy', 'energy_efficiency',
    'temp_diff', 'current_ratio', 'voltage_drop',
]

# Physics Engine outputs (5) — fed via dashed orange arrow in Fig 12.1
FEATURES_PHYS = [
    'SOH_phys', 'RUL_phys_norm', 'PDI',
    'thermal_stress', 'current_stress',
]

# Full fused feature set (19)
FEATURES_FUSED = FEATURES_14 + FEATURES_PHYS


def validate_features(df: pd.DataFrame, feature_list: list) -> list:
    """Return only features that actually exist in df."""
    available = [f for f in feature_list if f in df.columns]
    missing   = [f for f in feature_list if f not in df.columns]
    if missing:
        print(f"  [FeatureFusion] Warning: Missing columns {missing} — skipping.")
    return available


def fuse_features(df: pd.DataFrame,
                   use_physics: bool = True) -> tuple:
    """
    Build the fused feature matrix (X) and target matrix (y).

    Args:
        df          : DataFrame with both engineered + physics columns
        use_physics : If False, uses only 14 features (ablation study)

    Returns:
        X            : np.ndarray  (N × 19) or (N × 14)
        feature_cols : list of column names
    """
    if use_physics:
        feature_cols = validate_features(df, FEATURES_FUSED)
    else:
        feature_cols = validate_features(df, FEATURES_14)

    X = df[feature_cols].values.astype(np.float32)
    print(f"[FeatureFusion] Fused {X.shape[1]} features: "
          f"{len(FEATURES_14)} engineered + "
          f"{X.shape[1] - len(validate_features(df, FEATURES_14))} physics")
    return X, feature_cols


class FeatureScaler:
    """
    Scales feature matrix X and target matrix y for the hybrid ANN.

    Separate scalers for features (StandardScaler) and targets (MinMaxScaler).
    """

    def __init__(self):
        from sklearn.preprocessing import MinMaxScaler
        self.X_scaler = StandardScaler()
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'FeatureScaler':
        self.X_scaler.fit(X_train)
        self.y_scaler.fit(y_train)
        self.is_fitted = True
        print(f"[FeatureScaler] Fitted — X shape: {X_train.shape}, y shape: {y_train.shape}")
        return self

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return self.X_scaler.transform(X).astype(np.float32)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return self.y_scaler.transform(y).astype(np.float32)

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.y_scaler.inverse_transform(y_scaled).astype(np.float32)

    def fit_transform(self, X_train: np.ndarray, y_train: np.ndarray) -> tuple:
        self.fit(X_train, y_train)
        return self.transform_X(X_train), self.transform_y(y_train)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.X_scaler, os.path.join(directory, 'hybrid_X_scaler.pkl'))
        joblib.dump(self.y_scaler, os.path.join(directory, 'hybrid_y_scaler.pkl'))
        print(f"  [FeatureScaler] Saved → {directory}")

    def load(self, directory: str) -> 'FeatureScaler':
        self.X_scaler = joblib.load(os.path.join(directory, 'hybrid_X_scaler.pkl'))
        self.y_scaler = joblib.load(os.path.join(directory, 'hybrid_y_scaler.pkl'))
        self.is_fitted = True
        return self
