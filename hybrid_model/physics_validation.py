"""
Physics Validation Layer  —  PAS & DTCS
=========================================
Implements the "Physics Validation Layer (PAS, DTCS)" block from Fig 12.1
and the full "Validation Architecture" from Fig 11.1:

  ANN Predictions ŷ  ┐
                      ├─► Error Calculation Layer
  Ground Truth y     ─┘
                      │
                      ▼
          Metrics Layer (MAE, RMSE, R², PAS, DTCS)
                      │
                      ▼
          Performance Evaluation
                      │
                      ▼
          Dashboard & Reporting

New physics-specific metrics:

  PAS (Physics Adherence Score)
  ─────────────────────────────
  Measures how closely the ANN's SOH predictions align with the physics model:

      PAS = 1 − (1/N) Σ |SOH_pred_i − SOH_phys_i| / max(SOH_phys_i, ε)

  PAS ∈ [0, 1].  PAS = 1 → perfect physics adherence.

  DTCS (Degradation Trajectory Consistency Score)
  ────────────────────────────────────────────────
  Measures whether the predicted SOH trajectory monotonically decreases
  over cycles for each battery (physical expectation for battery aging):

      For each battery b:
          pairs_b = [(SOH_pred(n), SOH_pred(n+1)) for consecutive cycles]
          consistent_b = |{pairs where SOH(n+1) ≤ SOH(n)}| / |pairs_b|
      DTCS = mean(consistent_b) over all batteries

  DTCS ∈ [0, 1].  DTCS = 1 → always-decreasing SOH (ideal physics consistency).
"""

import numpy as np
import pandas as pd


# ── Error Calculation Layer (Fig 11.1) ────────────────────────────────────────

def error_calculation_layer(y_true: np.ndarray,
                             y_pred: np.ndarray) -> dict:
    """
    Compute raw error vectors for further metric computation.

    Returns:
        {'absolute_error', 'squared_error', 'relative_error'}  — all (N, 2) arrays
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ae     = np.abs(y_pred - y_true)
    se     = (y_pred - y_true) ** 2
    re     = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1e-8)

    return {'absolute_error': ae, 'squared_error': se, 'relative_error': re}


# ── Physics Adherence Score ────────────────────────────────────────────────────

def compute_pas(soh_pred:  np.ndarray,
                soh_phys:  np.ndarray,
                eps:       float = 1e-6) -> float:
    """
    Physics Adherence Score (PAS).

    PAS = 1 − mean( |SOH_pred − SOH_phys| / max(SOH_phys, ε) )

    Args:
        soh_pred : ANN-predicted SOH values  (N,)
        soh_phys : Physics-model SOH values  (N,)
        eps      : Numerical stability floor

    Returns:
        PAS ∈ [0, 1]
    """
    soh_pred = np.asarray(soh_pred, dtype=np.float64).flatten()
    soh_phys = np.asarray(soh_phys, dtype=np.float64).flatten()
    rel_err  = np.abs(soh_pred - soh_phys) / np.maximum(soh_phys, eps)
    pas      = float(1.0 - np.mean(np.clip(rel_err, 0, 1)))
    return max(0.0, min(1.0, pas))


# ── Degradation Trajectory Consistency Score ──────────────────────────────────

def compute_dtcs(soh_pred:    np.ndarray,
                  cycles:      np.ndarray,
                  battery_ids: np.ndarray) -> float:
    """
    Degradation Trajectory Consistency Score (DTCS).

    For each battery, checks what fraction of consecutive-cycle pairs
    satisfy SOH_pred(n+1) ≤ SOH_pred(n)  (monotonic decrease).

    Args:
        soh_pred    : ANN-predicted SOH   (N,)
        cycles      : Corresponding cycle numbers (N,)
        battery_ids : Corresponding battery IDs  (N,)

    Returns:
        DTCS ∈ [0, 1]
    """
    soh_pred    = np.asarray(soh_pred,    dtype=np.float64).flatten()
    cycles      = np.asarray(cycles,      dtype=np.float64).flatten()
    battery_ids = np.asarray(battery_ids).flatten()

    consistency_per_battery = []
    for bid in np.unique(battery_ids):
        mask  = battery_ids == bid
        c_b   = cycles[mask]
        s_b   = soh_pred[mask]
        order = np.argsort(c_b)
        s_b   = s_b[order]

        if len(s_b) < 2:
            continue

        diffs       = np.diff(s_b)            # SOH(n+1) - SOH(n)
        n_consistent = np.sum(diffs <= 1e-4)  # allow tiny tolerance
        consistency_per_battery.append(n_consistent / len(diffs))

    if not consistency_per_battery:
        return 0.0
    return float(np.mean(consistency_per_battery))


# ── Full Metrics Layer (Fig 11.1) ──────────────────────────────────────────────

def compute_metrics_layer(y_true:      np.ndarray,
                           y_pred:      np.ndarray,
                           y_phys_soh:  np.ndarray,
                           cycles:      np.ndarray,
                           battery_ids: np.ndarray) -> dict:
    """
    Full Metrics Layer:  MAE, RMSE, R², PAS, DTCS  for both SOH and RUL.

    Args:
        y_true      : (N, 2)  [SOH_true, RUL_true]
        y_pred      : (N, 2)  [SOH_pred, RUL_pred]
        y_phys_soh  : (N,)    SOH_phys for PAS calculation
        cycles      : (N,)    cycle numbers
        battery_ids : (N,)    battery IDs

    Returns:
        dict of all metrics for SOH and RUL, plus PAS and DTCS
    """
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                  r2_score, mean_absolute_percentage_error)

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    results = {}
    names   = ['SOH', 'RUL']
    for i, name in enumerate(names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        results[name] = {
            'Target'    : name,
            'MAE'       : round(mean_absolute_error(yt, yp), 4),
            'RMSE'      : round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
            'R2'        : round(r2_score(yt, yp), 4),
            'MAPE(%)'   : round(mean_absolute_percentage_error(yt, yp) * 100, 4),
            'Max Error' : round(float(np.max(np.abs(yt - yp))), 4),
        }

    # Physics-specific metrics
    pas  = compute_pas(y_pred[:, 0], y_phys_soh)
    dtcs = compute_dtcs(y_pred[:, 0], cycles, battery_ids)

    results['SOH']['PAS']  = round(pas,  4)
    results['SOH']['DTCS'] = round(dtcs, 4)
    results['RUL']['PAS']  = round(pas,  4)   # same physics consistency
    results['RUL']['DTCS'] = round(dtcs, 4)

    return results


# ── Performance Evaluation (Fig 11.1) ────────────────────────────────────────

def performance_evaluation(metrics_dict: dict) -> pd.DataFrame:
    """
    Format the metrics dict into a printable DataFrame for reporting.

    Returns:
        DataFrame with columns: Target, MAE, RMSE, R², MAPE(%), Max Error, PAS, DTCS
    """
    rows = list(metrics_dict.values())
    df   = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("  PHYSICS VALIDATION LAYER — Performance Evaluation")
    print("=" * 80)
    print(df.to_string(index=False))
    print(f"\n  PAS  (Physics Adherence Score)               = {df['PAS'].iloc[0]:.4f}  "
          f"[0=none, 1=perfect]")
    print(f"  DTCS (Degradation Trajectory Consistency)    = {df['DTCS'].iloc[0]:.4f}  "
          f"[0=none, 1=perfect]")
    print("=" * 80 + "\n")
    return df
