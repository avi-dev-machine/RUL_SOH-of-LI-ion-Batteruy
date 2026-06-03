"""
Metrics for the Hybrid Physics-Informed System
================================================
Provides the full Metrics Layer from the Validation Architecture (Fig 11.1):
    MAE, RMSE, R², MAPE, Max Error, PAS, DTCS

The compute_full_metrics() function is the main entry point,
implementing the complete Error Calculation → Metrics Layer chain.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)


def _single_target_metrics(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             name:   str) -> dict:
    """Standard regression metrics for one target."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return {
        'Target'    : name,
        'MAE'       : round(mean_absolute_error(y_true, y_pred), 4),
        'RMSE'      : round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        'R2'        : round(r2_score(y_true, y_pred), 4),
        'MAPE(%)'   : round(mean_absolute_percentage_error(y_true, y_pred) * 100, 4),
        'Max Error' : round(float(np.max(np.abs(y_true - y_pred))), 4),
    }


def compute_full_metrics(y_true:      np.ndarray,
                          y_pred:      np.ndarray,
                          y_phys_soh:  np.ndarray,
                          cycles:      np.ndarray,
                          battery_ids: np.ndarray) -> dict:
    """
    Full Metrics Layer (Fig 11.1): MAE, RMSE, R², MAPE, Max Error, PAS, DTCS.

    Args:
        y_true      : (N, 2)  [SOH_true, RUL_true]
        y_pred      : (N, 2)  [SOH_pred, RUL_pred]
        y_phys_soh  : (N,)    SOH from physics model
        cycles      : (N,)    cycle numbers
        battery_ids : (N,)    battery IDs

    Returns:
        dict {'SOH': {metrics}, 'RUL': {metrics}, 'summary': DataFrame}
    """
    from hybrid_model.physics_validation import compute_pas, compute_dtcs

    soh_metrics = _single_target_metrics(y_true[:, 0], y_pred[:, 0], 'SOH')
    rul_metrics = _single_target_metrics(y_true[:, 1], y_pred[:, 1], 'RUL')

    pas  = compute_pas(y_pred[:, 0], y_phys_soh)
    dtcs = compute_dtcs(y_pred[:, 0], cycles, battery_ids)

    soh_metrics['PAS']  = round(pas,  4)
    soh_metrics['DTCS'] = round(dtcs, 4)
    rul_metrics['PAS']  = round(pas,  4)
    rul_metrics['DTCS'] = round(dtcs, 4)

    summary = pd.DataFrame([soh_metrics, rul_metrics])
    return {'SOH': soh_metrics, 'RUL': rul_metrics, 'summary': summary}


def print_full_metrics(metrics: dict) -> None:
    """Pretty-print the metrics dict."""
    df = metrics.get('summary', pd.DataFrame(list(metrics.values())))
    print("\n" + "=" * 82)
    print("  HYBRID MODEL — FULL METRICS LAYER (MAE, RMSE, R², MAPE, PAS, DTCS)")
    print("=" * 82)
    print(df.to_string(index=False))
    print("=" * 82 + "\n")
