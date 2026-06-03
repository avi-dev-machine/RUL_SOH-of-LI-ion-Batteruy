"""
Hybrid Physics-Informed ANN — Main Pipeline
============================================
Implements the complete end-to-end system from Figure 12.1:

  NASA Battery Dataset (V, I, Q, T, R_int, N)
      ↓
  Data Preprocessing
      ↓
  Physics Engine  (Thermal + Stress Models)  ─────────────────┐
      ↓                                                         │ SOH_phys
  Feature Engineering  (14 features)                            │ RUL_phys
      ↓                                                         │ PDI
  Feature Fusion Layer                                          │
      ↓                                                         │
  Physics-Informed ANN  (Multi-task)  ◄──────────────────────┘
      ↓
  Prediction Layer  (ŷ_SOH, ŷ_RUL)
      ↓
  Physics Validation Layer  (PAS, DTCS)
      ↓
  Monitoring Dashboard

Usage:
    cd hybrid_model
    python main.py
"""

import os
import sys
import numpy as np
import pandas as pd

THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(THIS_DIR)
sys.path.insert(0, ROOT_DIR)   # allow  `from hybrid_model.xxx import ...`
sys.path.insert(0, THIS_DIR)

from data_loader         import (load_dataset, add_derived_features,
                                   split_data, FEATURE_14, PHYSICS_FEATURES)
from physics_engine      import PhysicsEngine
from feature_fusion      import fuse_features, FeatureScaler
from physics_informed_ann import PhysicsInformedBatteryANN
from physics_validation  import (error_calculation_layer,
                                  compute_pas, compute_dtcs,
                                  compute_metrics_layer,
                                  performance_evaluation)
from dashboard           import (create_monitoring_dashboard,
                                  plot_per_battery_validation)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(ROOT_DIR, "Battery_dataset (1).csv")
RESULTS_DIR = os.path.join(THIS_DIR, 'results')
MODELS_DIR  = os.path.join(THIS_DIR, 'saved_models')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 68)
    print("  HYBRID PHYSICS-INFORMED ANN — BATTERY SOH & RUL PREDICTION")
    print("=" * 68)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 · Data Preprocessing
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/7] Data Preprocessing …")
    df = load_dataset(DATA_PATH)
    df = add_derived_features(df)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 · Physics Engine  (Thermal + Stress Models)
    # Produces: SOH_phys, RUL_phys_norm, PDI, thermal_stress, current_stress
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/7] Physics Engine — fitting degradation curves …")
    physics_engine = PhysicsEngine()
    df = physics_engine.fit_transform(df)

    print("\n  Physics Engine parameters:")
    print(physics_engine.get_params_summary().to_string(index=False))

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 · Feature Fusion Layer  (14 engineered + 5 physics = 19 features)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3/7] Feature Fusion Layer …")
    X_fused, feature_cols = fuse_features(df, use_physics=True)
    y          = df[['SOH', 'RUL']].values.astype(np.float32)

    # Physics targets for the physics-informed loss
    phys_cols  = [c for c in PHYSICS_FEATURES if c in df.columns]
    y_phys_raw = df[phys_cols].values.astype(np.float32)

    print(f"  Fused feature matrix: {X_fused.shape}  "
          f"({len(FEATURE_14)} engineered + {len(phys_cols)} physics)")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 · Train / Val / Test split  +  Scaling
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[4/7] Splitting data and scaling …")
    data = split_data(df, feature_cols,
                       test_size=0.20, val_size=0.10, random_state=42)

    X_train, y_train = data['X_train'], data['y_train']
    X_val,   y_val   = data['X_val'],   data['y_val']
    X_test,  y_test  = data['X_test'],  data['y_test']
    y_phys_train     = data['y_phys_train']
    y_phys_val       = data['y_phys_val']
    y_phys_test      = data['y_phys_test']
    meta_test        = data['meta_test']

    scaler = FeatureScaler()
    X_train_s, y_train_s = scaler.fit_transform(X_train, y_train)
    X_val_s   = scaler.transform_X(X_val)
    y_val_s   = scaler.transform_y(y_val)
    X_test_s  = scaler.transform_X(X_test)

    # Scale physics targets (SOH_phys) to [0,1] same as y_scaler for SOH column
    # We use only SOH_phys (col 0) in physics loss
    y_phys_train_s = scaler.transform_y(
        np.column_stack([y_phys_train[:, 0],
                          np.zeros(len(y_phys_train))])
    )[:, :1]   # (N, 1)  — scaled SOH_phys only
    y_phys_val_s = scaler.transform_y(
        np.column_stack([y_phys_val[:, 0],
                          np.zeros(len(y_phys_val))])
    )[:, :1]

    scaler.save(MODELS_DIR)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5 · Physics-Informed ANN Training
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[5/7] Training Physics-Informed ANN (Multi-task) …")
    ann = PhysicsInformedBatteryANN(
        input_dim     = X_train_s.shape[1],
        trunk_units   = [256, 128, 64],
        head_units    = [32, 16],
        dropout_rate  = 0.20,
        weight_decay  = 1e-4,
        learning_rate = 1e-3,
        lambda_data   = 1.00,   # data-fidelity weight
        lambda_phys   = 0.30,   # physics-adherence weight (orange dashed arrow)
        lambda_mono   = 0.10,   # monotonicity constraint weight
    )

    history = ann.fit(
        X_train_s, y_train_s, y_phys_train_s,
        X_val_s,   y_val_s,   y_phys_val_s,
        epochs         = 300,
        batch_size     = 32,
        patience       = 30,
        checkpoint_dir = MODELS_DIR,
    )
    ann.save(MODELS_DIR)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6 · Prediction Layer  (ŷ_SOH, ŷ_RUL)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[6/7] Prediction Layer — generating forecasts …")
    y_pred_scaled = ann.predict(X_test_s)
    y_pred        = scaler.inverse_y(y_pred_scaled)   # back to original scale

    print(f"  Predictions generated: {y_pred.shape} (SOH, RUL) for {len(y_pred)} test samples")

    # Annotate df_full with hybrid predictions for dashboard
    df_full = df.copy()
    df_full['SOH_hybrid'] = np.nan
    test_idx              = meta_test.index   # integer positions in meta_test
    # We need to map test samples back to df_full rows
    df_all_pred = ann.predict(scaler.transform_X(X_fused))
    df_all_pred_orig = scaler.inverse_y(df_all_pred)
    df_full['SOH_hybrid'] = df_all_pred_orig[:, 0]
    df_full['RUL_hybrid'] = df_all_pred_orig[:, 1]

    # ─────────────────────────────────────────────────────────────────────────
    # Step 7 · Physics Validation Layer  (PAS, DTCS)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[7/7] Physics Validation Layer — PAS & DTCS …")

    # Error Calculation Layer (Fig 11.1)
    errors = error_calculation_layer(y_test, y_pred)
    print(f"  MAE  (SOH): {errors['absolute_error'][:, 0].mean():.4f}")
    print(f"  MAE  (RUL): {errors['absolute_error'][:, 1].mean():.4f}")

    # Physics Adherence Score
    soh_phys_test = y_phys_test[:, 0]   # SOH_phys for test samples
    pas_overall   = compute_pas(y_pred[:, 0], soh_phys_test)

    # DTCS
    cycles_test   = meta_test['cycle'].values
    bids_test     = meta_test['battery_id'].values
    dtcs_overall  = compute_dtcs(y_pred[:, 0], cycles_test, bids_test)

    # Per-battery PAS and DTCS
    pas_per_bat  = {}
    dtcs_per_bat = {}
    for bid in sorted(df['battery_id'].unique()):
        mask = bids_test == bid
        if mask.sum() < 2:
            continue
        pas_per_bat[bid]  = compute_pas(y_pred[mask, 0], soh_phys_test[mask])
        dtcs_per_bat[bid] = compute_dtcs(y_pred[mask, 0],
                                          cycles_test[mask],
                                          bids_test[mask])

    print(f"\n  PAS  (Physics Adherence Score) = {pas_overall:.4f}  [0-1, higher=better]")
    print(f"  DTCS (Trajectory Consistency)  = {dtcs_overall:.4f}  [0-1, higher=better]")
    for bid in sorted(pas_per_bat):
        print(f"    {bid}: PAS={pas_per_bat[bid]:.3f}  DTCS={dtcs_per_bat[bid]:.3f}")

    # Full Metrics Layer (Fig 11.1)
    metrics = compute_metrics_layer(
        y_test, y_pred, soh_phys_test, cycles_test, bids_test
    )
    metrics_df = performance_evaluation(metrics)

    # Save metrics CSV
    csv_path = os.path.join(RESULTS_DIR, 'hybrid_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Monitoring Dashboard  (Fig 12.1 final block)
    # ─────────────────────────────────────────────────────────────────────────
    print("\nBuilding Monitoring Dashboard …")

    # Comparison: try to load standalone ANN metrics if available
    comparison = {
        'Hybrid ANN': {
            'RMSE_SOH': metrics['SOH']['RMSE'],
            'MAE_SOH' : metrics['SOH']['MAE'],
            'RMSE_RUL': metrics['RUL']['RMSE'],
            'MAE_RUL' : metrics['RUL']['MAE'],
        },
        'Standalone ANN': {
            'RMSE_SOH': 2.39,   # from ann_model results
            'MAE_SOH' : 1.80,
            'RMSE_RUL': 15.05,
            'MAE_RUL' : 12.90,
        }
    }

    create_monitoring_dashboard(
        df_full    = df_full,
        y_true     = y_test,
        y_pred     = y_pred,
        history    = history,
        pas        = pas_overall,
        dtcs       = dtcs_overall,
        metrics_df = metrics_df,
        comparison = comparison,
    )

    plot_per_battery_validation(df_full, pas_per_bat, dtcs_per_bat)

    # ─────────────────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  HYBRID SYSTEM — FINAL RESULTS SUMMARY")
    print("=" * 68)
    print(f"  Features used          : {len(feature_cols)}"
          f" (14 engineered + {len(feature_cols)-14} physics)")
    print(f"  Training samples       : {len(X_train)}")
    print(f"  Test samples           : {len(X_test)}")
    print()
    soh_m = metrics['SOH']
    rul_m = metrics['RUL']
    print(f"  SOH | RMSE: {soh_m['RMSE']:.4f}  MAE: {soh_m['MAE']:.4f}  "
          f"R²: {soh_m['R2']:.4f}")
    print(f"  RUL | RMSE: {rul_m['RMSE']:.4f}  MAE: {rul_m['MAE']:.4f}  "
          f"R²: {rul_m['R2']:.4f}")
    print()
    print(f"  PAS  (Physics Adherence Score)     = {pas_overall:.4f}")
    print(f"  DTCS (Trajectory Consistency Score)= {dtcs_overall:.4f}")
    print()
    print(f"  Results saved → {RESULTS_DIR}")
    print(f"  Models  saved → {MODELS_DIR}")
    print("=" * 68 + "\n")


if __name__ == '__main__':
    main()
