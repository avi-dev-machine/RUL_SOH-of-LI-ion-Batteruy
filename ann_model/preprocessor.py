"""
Data Preprocessor for ANN Battery Model
=========================================
Handles feature scaling and target normalization / inverse-transformation.

Feature scaler  : StandardScaler (zero-mean, unit-variance) — best for ANN training
Target scaler   : MinMaxScaler [0, 1]                        — keeps outputs bounded
"""

import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class BatteryPreprocessor:
    """
    Fit-once / transform-many preprocessor for battery feature matrices
    and multi-output target arrays.

    Workflow:
        1. preprocessor.fit(X_train, y_train)
        2. X_train_s, y_train_s = preprocessor.transform(X_train, y_train)
        3. X_val_s  = preprocessor.transform_X(X_val)
        4. y_pred_original = preprocessor.inverse_y(y_pred_scaled)
    """

    def __init__(self,
                 feature_scaler: str = 'standard',
                 target_scaler:  str = 'minmax'):
        """
        Args:
            feature_scaler : 'standard' (StandardScaler) or 'minmax'
            target_scaler  : 'minmax' (MinMaxScaler) or 'standard'
        """
        self._feature_scaler_type = feature_scaler
        self._target_scaler_type  = target_scaler

        self.feature_scaler = (StandardScaler() if feature_scaler == 'standard'
                               else MinMaxScaler())
        self.target_scaler  = (MinMaxScaler()   if target_scaler  == 'minmax'
                               else StandardScaler())
        self.is_fitted = False

    # ── Fitting ────────────────────────────────────────────────────
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray) -> 'BatteryPreprocessor':
        """Fit both scalers on training data only."""
        self.feature_scaler.fit(X_train)
        self.target_scaler.fit(y_train)
        self.is_fitted = True
        print(f"✅ Preprocessor fitted  "
              f"(X: {self._feature_scaler_type}Scaler, "
              f"y: {self._target_scaler_type}Scaler)")
        return self

    # ── Transforms ─────────────────────────────────────────────────
    def transform_X(self, X: np.ndarray) -> np.ndarray:
        """Scale feature matrix."""
        self._check_fitted()
        return self.feature_scaler.transform(X).astype(np.float32)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        """Scale target matrix."""
        self._check_fitted()
        return self.target_scaler.transform(y).astype(np.float32)

    def transform(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Scale both X and y."""
        return self.transform_X(X), self.transform_y(y)

    def fit_transform(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray) -> tuple:
        """Fit and immediately transform (for training set only)."""
        self.fit(X_train, y_train)
        return self.transform(X_train, y_train)

    # ── Inverse transform ──────────────────────────────────────────
    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse-transform scaled predictions back to original scale."""
        self._check_fitted()
        return self.target_scaler.inverse_transform(y_scaled).astype(np.float32)

    # ── Persistence ────────────────────────────────────────────────
    def save(self, directory: str) -> None:
        """Persist scalers to disk."""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.feature_scaler,
                    os.path.join(directory, 'feature_scaler.pkl'))
        joblib.dump(self.target_scaler,
                    os.path.join(directory, 'target_scaler.pkl'))
        print(f"  💾 Preprocessor saved → {directory}")

    def load(self, directory: str) -> 'BatteryPreprocessor':
        """Load persisted scalers from disk."""
        self.feature_scaler = joblib.load(
            os.path.join(directory, 'feature_scaler.pkl'))
        self.target_scaler  = joblib.load(
            os.path.join(directory, 'target_scaler.pkl'))
        self.is_fitted = True
        return self

    # ── Helper ─────────────────────────────────────────────────────
    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Call fit() or load() before transforming data.")
