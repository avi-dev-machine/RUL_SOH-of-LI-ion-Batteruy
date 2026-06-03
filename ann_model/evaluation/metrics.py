"""
Evaluation Metrics for ANN Battery Model
==========================================
MAE, RMSE, R², MAPE and Max-Error for both SOH and RUL targets.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    target_name: str = '') -> dict:
    """
    Compute regression metrics for a single target.

    Args:
        y_true      : Ground-truth array  (N,)
        y_pred      : Prediction array    (N,)
        target_name : Label string ('SOH' or 'RUL')

    Returns:
        dict: {Target, MAE, RMSE, R2, MAPE(%), Max_Error}
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    return {
        'Target'    : target_name,
        'MAE'       : round(mean_absolute_error(y_true, y_pred), 4),
        'RMSE'      : round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        'R2'        : round(r2_score(y_true, y_pred), 4),
        'MAPE (%)'  : round(mean_absolute_percentage_error(y_true, y_pred) * 100.0, 4),
        'Max Error' : round(float(np.max(np.abs(y_true - y_pred))), 4),
    }


def compute_all_metrics(y_true: np.ndarray,
                         y_pred: np.ndarray) -> dict:
    """
    Compute metrics for both SOH (col 0) and RUL (col 1) simultaneously.

    Args:
        y_true : shape (N, 2)  — columns: [SOH, RUL]
        y_pred : shape (N, 2)  — columns: [SOH_pred, RUL_pred]

    Returns:
        {'SOH': metrics_dict, 'RUL': metrics_dict}
    """
    return {
        'SOH': compute_metrics(y_true[:, 0], y_pred[:, 0], 'SOH'),
        'RUL': compute_metrics(y_true[:, 1], y_pred[:, 1], 'RUL'),
    }


def print_metrics(metrics_dict: dict) -> None:
    """Pretty-print a {target: metrics} dict as a table."""
    rows = list(metrics_dict.values())
    df   = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("  ANN MODEL — EVALUATION RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70 + "\n")
