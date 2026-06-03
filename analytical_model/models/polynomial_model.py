"""
Polynomial Regression Model for Battery Degradation
======================================================
Extends linear regression with polynomial feature expansion.
Uses Ridge regularization to prevent overfitting from high-degree terms.

Typical use:
    - degree=2 captures nonlinear capacity fade
    - degree=3 captures inflection in late-life degradation
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
import joblib


class PolynomialDegradationModel:
    """
    Polynomial regression with Ridge regularization for SOH and RUL prediction.

    Pipeline: StandardScaler → PolynomialFeatures → Ridge
    """

    def __init__(self, degree: int = 2, alpha: float = 1.0):
        """
        Args:
            degree : Polynomial expansion degree (2 or 3 recommended)
            alpha  : Ridge regularization strength
        """
        self.degree    = degree
        self.alpha     = alpha
        self.pipelines = {}
        self.is_fitted = False

    # ── Private ────────────────────────────────────────────────────
    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('poly',   PolynomialFeatures(degree=self.degree,
                                          include_bias=False,
                                          interaction_only=False)),
            ('model',  Ridge(alpha=self.alpha))
        ])

    # ── Public API ─────────────────────────────────────────────────
    def fit(self,
            X: pd.DataFrame,
            y_soh: pd.Series,
            y_rul: pd.Series) -> 'PolynomialDegradationModel':
        """Fit polynomial pipelines for SOH and RUL."""
        for target, y in [('SOH', y_soh), ('RUL', y_rul)]:
            pipeline = self._build_pipeline()
            pipeline.fit(X, y)
            self.pipelines[target] = pipeline

            # Report approximate number of features after expansion
            n_feat = pipeline.named_steps['poly'].n_output_features_
            print(f"    ✔ Poly (deg={self.degree}, α={self.alpha}) fitted for "
                  f"{target}  [{n_feat} expanded features]")

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> dict:
        """Predict SOH and RUL."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        return {
            'SOH': self.pipelines['SOH'].predict(X),
            'RUL': self.pipelines['RUL'].predict(X)
        }

    def cross_validate(self,
                        X: pd.DataFrame,
                        y: pd.Series,
                        target: str = 'SOH',
                        cv: int = 5) -> dict:
        """K-Fold cross-validation."""
        pipeline    = self._build_pipeline()
        kf          = KFold(n_splits=cv, shuffle=True, random_state=42)
        r2_scores   = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
        rmse_scores = np.sqrt(-cross_val_score(
            pipeline, X, y, cv=kf, scoring='neg_mean_squared_error'))

        return {
            'Model'              : f'Poly(deg={self.degree})',
            'Target'             : target,
            'CV-R2 (mean±std)'   : f"{r2_scores.mean():.4f} ± {r2_scores.std():.4f}",
            'CV-RMSE (mean±std)' : f"{rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}"
        }

    def save(self, directory: str) -> None:
        """Save pipelines to disk."""
        os.makedirs(directory, exist_ok=True)
        for target, pipeline in self.pipelines.items():
            path = os.path.join(directory, f"poly_deg{self.degree}_{target.lower()}.pkl")
            joblib.dump(pipeline, path)
            print(f"    💾 Saved {target} pipeline → {path}")

    def load(self, directory: str) -> None:
        """Load saved pipelines from disk."""
        for target in ['SOH', 'RUL']:
            path = os.path.join(directory, f"poly_deg{self.degree}_{target.lower()}.pkl")
            if os.path.exists(path):
                self.pipelines[target] = joblib.load(path)
        self.is_fitted = bool(self.pipelines)
