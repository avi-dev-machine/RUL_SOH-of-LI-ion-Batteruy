"""
Linear Regression Models for Battery SOH and RUL Prediction
=============================================================
Implements Linear, Ridge, and Lasso regression with StandardScaler pipelines.
Supports per-target fitting (SOH and RUL independently).
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


class LinearDegradationModel:
    """
    Linear regression-based battery degradation model.

    Supported variants:
        'linear'     : Ordinary Least Squares
        'ridge'      : Ridge (L2) regularization
        'lasso'      : Lasso (L1) regularization
        'elasticnet' : ElasticNet (L1 + L2) regularization
    """

    def __init__(self, model_type: str = 'ridge', alpha: float = 1.0):
        """
        Args:
            model_type : One of 'linear', 'ridge', 'lasso', 'elasticnet'
            alpha      : Regularization strength (ignored for 'linear')
        """
        self.model_type = model_type
        self.alpha      = alpha
        self.pipelines  = {}       # {'SOH': pipeline, 'RUL': pipeline}
        self.is_fitted  = False

    # ── Private helpers ────────────────────────────────────────────
    def _build_estimator(self):
        if self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'ridge':
            return Ridge(alpha=self.alpha)
        elif self.model_type == 'lasso':
            return Lasso(alpha=self.alpha, max_iter=10_000)
        elif self.model_type == 'elasticnet':
            return ElasticNet(alpha=self.alpha, l1_ratio=0.5, max_iter=10_000)
        else:
            raise ValueError(f"Unknown model_type '{self.model_type}'. "
                             f"Choose from: linear, ridge, lasso, elasticnet")

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model',  self._build_estimator())
        ])

    # ── Public API ─────────────────────────────────────────────────
    def fit(self,
            X: pd.DataFrame,
            y_soh: pd.Series,
            y_rul: pd.Series) -> 'LinearDegradationModel':
        """
        Fit separate pipelines for SOH and RUL.

        Args:
            X     : Feature matrix (N × F)
            y_soh : SOH target vector (N,)
            y_rul : RUL target vector (N,)
        """
        for target, y in [('SOH', y_soh), ('RUL', y_rul)]:
            pipeline = self._build_pipeline()
            pipeline.fit(X, y)
            self.pipelines[target] = pipeline
            print(f"    ✔ {self.model_type.capitalize():12s} fitted for {target}")

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> dict:
        """
        Predict SOH and RUL.

        Returns:
            {'SOH': np.ndarray, 'RUL': np.ndarray}
        """
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
        """
        K-Fold cross-validation on the pipeline.

        Returns:
            {'R2_mean', 'R2_std', 'RMSE_mean', 'RMSE_std'}
        """
        pipeline    = self._build_pipeline()
        kf          = KFold(n_splits=cv, shuffle=True, random_state=42)
        r2_scores   = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
        rmse_scores = np.sqrt(-cross_val_score(
            pipeline, X, y, cv=kf, scoring='neg_mean_squared_error'))

        result = {
            'Model'    : self.model_type,
            'Target'   : target,
            f'CV-R2 (mean±std)'  : f"{r2_scores.mean():.4f} ± {r2_scores.std():.4f}",
            f'CV-RMSE (mean±std)': f"{rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}"
        }
        return result

    def get_coefficients(self, feature_names: list) -> dict:
        """Return feature coefficients for each target."""
        result = {}
        for target, pipeline in self.pipelines.items():
            estimator = pipeline.named_steps['model']
            if hasattr(estimator, 'coef_'):
                result[target] = dict(zip(feature_names, estimator.coef_))
        return result

    def get_intercept(self) -> dict:
        """Return intercepts."""
        return {t: p.named_steps['model'].intercept_
                for t, p in self.pipelines.items()
                if hasattr(p.named_steps['model'], 'intercept_')}

    def save(self, directory: str) -> None:
        """Persist pipelines to disk using joblib."""
        os.makedirs(directory, exist_ok=True)
        for target, pipeline in self.pipelines.items():
            path = os.path.join(directory, f"{self.model_type}_{target.lower()}.pkl")
            joblib.dump(pipeline, path)
            print(f"    💾 Saved {target} pipeline → {path}")

    def load(self, directory: str) -> None:
        """Reload saved pipelines from disk."""
        for target in ['SOH', 'RUL']:
            path = os.path.join(directory, f"{self.model_type}_{target.lower()}.pkl")
            if os.path.exists(path):
                self.pipelines[target] = joblib.load(path)
        self.is_fitted = bool(self.pipelines)
