"""
Evaluation Metrics for Battery Prediction Models
==================================================
Provides MAE, RMSE, R², MAPE, and Max-Error computation
along with pretty-printing and model comparison utilities.
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
    Compute a full set of regression evaluation metrics.

    Args:
        y_true      : Ground-truth values
        y_pred      : Model predictions
        target_name : Label for the target (e.g. 'SOH', 'RUL')

    Returns:
        dict with keys: Target, MAE, RMSE, R2, MAPE(%), Max_Error
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae      = mean_absolute_error(y_true, y_pred)
    rmse     = np.sqrt(mean_squared_error(y_true, y_pred))
    r2       = r2_score(y_true, y_pred)
    mape     = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    max_err  = float(np.max(np.abs(y_true - y_pred)))

    return {
        'Target'    : target_name,
        'MAE'       : round(mae,     4),
        'RMSE'      : round(rmse,    4),
        'R2'        : round(r2,      4),
        'MAPE (%)'  : round(mape,    4),
        'Max Error' : round(max_err, 4),
    }


def print_metrics_table(metrics_list: list) -> None:
    """
    Print a formatted table of metrics.

    Args:
        metrics_list : list of dicts returned by compute_metrics()
    """
    df = pd.DataFrame(metrics_list)
    print("\n" + "=" * 75)
    print("  EVALUATION RESULTS")
    print("=" * 75)
    print(df.to_string(index=False))
    print("=" * 75 + "\n")


def compare_models(results: dict) -> pd.DataFrame:
    """
    Flatten a nested results dict into a comparison DataFrame.

    Args:
        results : {model_name: {'SOH': metrics_dict, 'RUL': metrics_dict}}

    Returns:
        DataFrame with columns: Model, Target, MAE, RMSE, R2, MAPE(%), Max Error
    """
    rows = []
    for model_name, targets in results.items():
        for target_name, metrics in targets.items():
            rows.append({'Model': model_name, **metrics})

    df = pd.DataFrame(rows)

    print("\n" + "=" * 85)
    print("  MODEL COMPARISON")
    print("=" * 85)
    print(df.to_string(index=False))
    print("=" * 85 + "\n")

    return df
