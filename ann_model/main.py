"""
ANN Model — Main Training & Evaluation Script (PyTorch)
========================================================
Trains and compares two ANN architectures for battery
State-of-Health (SOH) and Remaining Useful Life (RUL) prediction.

Architectures:
    1. Standard ANN  — deep feed-forward MLP  (256→128→64→32→2)
    2. Multi-Task ANN — shared trunk + SOH/RUL heads

Pipeline:
  Load → Feature Engineering → Scale → Train → Evaluate → Visualize → Save

Usage:
    cd ann_model
    python main.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from data_loader  import load_dataset, prepare_data, add_derived_features, get_feature_list
from preprocessor import BatteryPreprocessor
from models       import BatteryANN
from evaluation   import compute_all_metrics, print_metrics
from trainer      import (plot_training_history, plot_predictions,
                           plot_timeline, plot_error_vs_cycle)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(THIS_DIR), "Battery_dataset (1).csv")
RESULTS_DIR = os.path.join(THIS_DIR, 'results')
MODELS_DIR  = os.path.join(THIS_DIR, 'saved_models')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
def run_architecture(arch:        str,
                      X_train_s:  np.ndarray,
                      y_train_s:  np.ndarray,
                      X_val_s:    np.ndarray,
                      y_val_s:    np.ndarray,
                      X_test_s:   np.ndarray,
                      y_test:     np.ndarray,
                      prep:       BatteryPreprocessor,
                      feat_names: list,
                      targ_names: list,
                      cycles_test: np.ndarray) -> dict:
    """Train, evaluate, and visualise one architecture. Returns metrics dict."""

    print(f"\n{'═'*60}")
    print(f"  Architecture: {arch.upper()}")
    print(f"{'═'*60}")

    ann = BatteryANN(
        input_dim     = X_train_s.shape[1],
        architecture  = arch,
        hidden_units  = [256, 128, 64, 32],
        dropout_rate  = 0.20,
        weight_decay  = 1e-4,
        learning_rate = 1e-3,
    )

    history = ann.fit(
        X_train_s, y_train_s,
        X_val_s,   y_val_s,
        epochs      = 300,
        batch_size  = 32,
        patience    = 30,
        checkpoint_dir = MODELS_DIR,
    )
    ann.save(MODELS_DIR, tag='final')

    # ── Inference ──────────────────────────────────────────────────
    y_pred_scaled = ann.predict(X_test_s)
    y_pred        = prep.inverse_y(y_pred_scaled)   # original scale
    y_true        = y_test

    # ── Metrics ────────────────────────────────────────────────────
    metrics = compute_all_metrics(y_true, y_pred)
    print(f"\n  [{arch}] Evaluation:")
    print_metrics(metrics)

    # ── Plots ──────────────────────────────────────────────────────
    plot_training_history(history, arch=arch)
    plot_predictions(y_true, y_pred, targ_names, arch=arch)
    plot_timeline(y_true, y_pred, targ_names, n_samples=120, arch=arch)
    plot_error_vs_cycle(y_true, y_pred, cycles_test, 'SOH', 0, arch=arch)
    plot_error_vs_cycle(y_true, y_pred, cycles_test, 'RUL', 1, arch=arch)

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═" * 65)
    print("  🔋  BATTERY ANN MODEL  —  Training & Evaluation  (PyTorch)")
    print("═" * 65)

    # ── 1. LOAD ────────────────────────────────────────────────────
    print("\n[Step 1/6]  Loading dataset …")
    df = load_dataset(DATA_PATH)

    # ── 2. PREPARE DATA ────────────────────────────────────────────
    print("\n[Step 2/6]  Feature engineering & data splitting …")
    data = prepare_data(df, use_derived=True,
                         test_size=0.20, val_size=0.10, random_state=42)

    X_train, y_train = data['X_train'], data['y_train']
    X_val,   y_val   = data['X_val'],   data['y_val']
    X_test,  y_test  = data['X_test'],  data['y_test']
    feat_names       = data['feature_names']
    targ_names       = data['target_names']

    # Build cycle metadata for test set (for error-vs-cycle plots)
    df_eng     = add_derived_features(df)
    features   = [f for f in get_feature_list(True) if f in df_eng.columns]
    X_all      = df_eng[features].values.astype(np.float32)
    y_all      = df_eng[['SOH', 'RUL']].values.astype(np.float32)
    c_all      = df_eng['cycle'].values.astype(np.float32)
    _, _, _, _, c_temp, _ = train_test_split(
        X_all, y_all, c_all, test_size=0.30, random_state=42, shuffle=True
    )
    _, cycles_test = train_test_split(c_temp, test_size=0.667, random_state=42)

    # ── 3. PREPROCESS ──────────────────────────────────────────────
    print("\n[Step 3/6]  Scaling features and targets …")
    prep = BatteryPreprocessor(feature_scaler='standard', target_scaler='minmax')

    X_train_s, y_train_s = prep.fit_transform(X_train, y_train)
    X_val_s   = prep.transform_X(X_val)
    y_val_s   = prep.transform_y(y_val)
    X_test_s  = prep.transform_X(X_test)
    prep.save(MODELS_DIR)

    # ── 4. TRAIN BOTH ARCHITECTURES ────────────────────────────────
    print("\n[Step 4/6]  Training ANN architectures …")
    all_metrics = {}
    for arch in ('standard', 'multitask'):
        all_metrics[arch] = run_architecture(
            arch        = arch,
            X_train_s   = X_train_s,
            y_train_s   = y_train_s,
            X_val_s     = X_val_s,
            y_val_s     = y_val_s,
            X_test_s    = X_test_s,
            y_test      = y_test,
            prep        = prep,
            feat_names  = feat_names,
            targ_names  = targ_names,
            cycles_test = cycles_test,
        )

    # ── 5. COMPARISON TABLE ────────────────────────────────────────
    print("\n[Step 5/6]  Comparing architectures …")
    rows = []
    for arch, targets in all_metrics.items():
        for _, m in targets.items():
            rows.append({'Architecture': arch, **m})
    cmp_df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("  ARCHITECTURE COMPARISON")
    print("=" * 78)
    print(cmp_df.to_string(index=False))
    print("=" * 78)

    # ── 6. SAVE ────────────────────────────────────────────────────
    print("\n[Step 6/6]  Saving metrics …")
    csv_path = os.path.join(RESULTS_DIR, 'ann_metrics.csv')
    cmp_df.to_csv(csv_path, index=False)
    print(f"  💾 Metrics CSV → {csv_path}")

    # ── FINAL SUMMARY ──────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  ✅  ANN model pipeline complete!")
    print(f"  📁  Plots  → {RESULTS_DIR}")
    print(f"  📁  Models → {MODELS_DIR}")
    for arch, targets in all_metrics.items():
        soh = targets['SOH']
        rul = targets['RUL']
        print(f"\n  [{arch.upper()}]")
        print(f"    SOH → RMSE: {soh['RMSE']:.4f}  R²: {soh['R2']:.4f}  MAE: {soh['MAE']:.4f}")
        print(f"    RUL → RMSE: {rul['RMSE']:.4f}  R²: {rul['R2']:.4f}  MAE: {rul['MAE']:.4f}")
    print("═" * 65 + "\n")


if __name__ == '__main__':
    main()
