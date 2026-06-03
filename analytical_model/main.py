"""
Analytical Model — Main Training & Evaluation Script
======================================================
Trains and compares multiple analytical/statistical models for
battery State-of-Health (SOH) and Remaining Useful Life (RUL) prediction.

Models evaluated:
    1. Linear Regression
    2. Ridge Regression  (α = 1.0)
    3. Lasso Regression  (α = 0.01)
    4. Polynomial Regression (degree = 2)
    5. Polynomial Regression (degree = 3)
    6. Exponential Degradation Model  (physics-based, per battery)

Usage:
    cd analytical_model
    python main.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Resolve imports ────────────────────────────────────────────────
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from data_loader import load_dataset, clean_dataset, describe_dataset
from feature_engineering import compute_derived_features, get_all_features
from eda import run_eda
from models import LinearDegradationModel, PolynomialDegradationModel, ExponentialDegradationModel
from evaluation import compute_metrics, print_metrics_table, compare_models
from visualization import (
    plot_predictions_vs_actual,
    plot_degradation_fit,
    plot_feature_importance,
    plot_model_comparison,
)

DATA_PATH    = os.path.join(os.path.dirname(THIS_DIR), "Battery_dataset (1).csv")
RESULTS_DIR  = os.path.join(THIS_DIR, 'results')
MODELS_DIR   = os.path.join(THIS_DIR, 'saved_models')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═" * 65)
    print("  🔋  BATTERY ANALYTICAL MODEL  —  Training & Evaluation")
    print("═" * 65)

    # ── 1. LOAD & CLEAN DATA ──────────────────────────────────────
    print("\n[Step 1/7]  Loading and cleaning dataset …")
    df = load_dataset(DATA_PATH)
    df = clean_dataset(df)
    describe_dataset(df)

    # ── 2. EDA ────────────────────────────────────────────────────
    print("[Step 2/7]  Running Exploratory Data Analysis …")
    run_eda(df)

    # ── 3. FEATURE ENGINEERING ────────────────────────────────────
    print("[Step 3/7]  Engineering features …")
    df           = compute_derived_features(df)
    feature_cols = get_all_features(with_derived=True)
    feature_cols = [c for c in feature_cols if c in df.columns]

    X     = df[feature_cols]
    y_soh = df['SOH']
    y_rul = df['RUL']

    X_train, X_test, y_soh_train, y_soh_test, y_rul_train, y_rul_test = (
        train_test_split(X, y_soh, y_rul, test_size=0.20,
                          random_state=42, shuffle=True)
    )
    print(f"  Features  : {len(feature_cols)}")
    print(f"  Train     : {len(X_train)} samples")
    print(f"  Test      : {len(X_test)} samples")

    all_results = {}

    # ── 4. LINEAR REGRESSION ──────────────────────────────────────
    print("\n[Step 4/7]  Training analytical models …\n")

    print("  ─── Linear Regression ───────────────────────────────")
    lin = LinearDegradationModel(model_type='linear')
    lin.fit(X_train, y_soh_train, y_rul_train)
    lin_pred = lin.predict(X_test)
    all_results['Linear Regression'] = {
        'SOH': compute_metrics(y_soh_test, lin_pred['SOH'], 'SOH'),
        'RUL': compute_metrics(y_rul_test, lin_pred['RUL'], 'RUL'),
    }
    plot_predictions_vs_actual(y_soh_test, lin_pred['SOH'], 'SOH', 'Linear Regression')
    plot_predictions_vs_actual(y_rul_test, lin_pred['RUL'], 'RUL', 'Linear Regression')
    plot_feature_importance(lin.get_coefficients(feature_cols), 'Linear Regression')
    lin.save(MODELS_DIR)

    # ── 5. RIDGE REGRESSION ───────────────────────────────────────
    print("\n  ─── Ridge Regression (α=1.0) ────────────────────────")
    ridge = LinearDegradationModel(model_type='ridge', alpha=1.0)
    ridge.fit(X_train, y_soh_train, y_rul_train)
    ridge_pred = ridge.predict(X_test)
    all_results['Ridge Regression'] = {
        'SOH': compute_metrics(y_soh_test, ridge_pred['SOH'], 'SOH'),
        'RUL': compute_metrics(y_rul_test, ridge_pred['RUL'], 'RUL'),
    }
    plot_predictions_vs_actual(y_soh_test, ridge_pred['SOH'], 'SOH', 'Ridge Regression')
    plot_predictions_vs_actual(y_rul_test, ridge_pred['RUL'], 'RUL', 'Ridge Regression')
    ridge.save(MODELS_DIR)

    # ── 6. LASSO REGRESSION ───────────────────────────────────────
    print("\n  ─── Lasso Regression (α=0.01) ───────────────────────")
    lasso = LinearDegradationModel(model_type='lasso', alpha=0.01)
    lasso.fit(X_train, y_soh_train, y_rul_train)
    lasso_pred = lasso.predict(X_test)
    all_results['Lasso Regression'] = {
        'SOH': compute_metrics(y_soh_test, lasso_pred['SOH'], 'SOH'),
        'RUL': compute_metrics(y_rul_test, lasso_pred['RUL'], 'RUL'),
    }
    lasso.save(MODELS_DIR)

    # ── 7. POLYNOMIAL (deg=2) ─────────────────────────────────────
    print("\n  ─── Polynomial Regression  degree=2 ─────────────────")
    poly2 = PolynomialDegradationModel(degree=2, alpha=1.0)
    poly2.fit(X_train, y_soh_train, y_rul_train)
    poly2_pred = poly2.predict(X_test)
    all_results['Polynomial (deg=2)'] = {
        'SOH': compute_metrics(y_soh_test, poly2_pred['SOH'], 'SOH'),
        'RUL': compute_metrics(y_rul_test, poly2_pred['RUL'], 'RUL'),
    }
    plot_predictions_vs_actual(y_soh_test, poly2_pred['SOH'], 'SOH', 'Polynomial (deg=2)')
    plot_predictions_vs_actual(y_rul_test, poly2_pred['RUL'], 'RUL', 'Polynomial (deg=2)')
    poly2.save(MODELS_DIR)

    # ── 8. POLYNOMIAL (deg=3) ─────────────────────────────────────
    print("\n  ─── Polynomial Regression  degree=3 ─────────────────")
    poly3 = PolynomialDegradationModel(degree=3, alpha=10.0)
    poly3.fit(X_train, y_soh_train, y_rul_train)
    poly3_pred = poly3.predict(X_test)
    all_results['Polynomial (deg=3)'] = {
        'SOH': compute_metrics(y_soh_test, poly3_pred['SOH'], 'SOH'),
        'RUL': compute_metrics(y_rul_test, poly3_pred['RUL'], 'RUL'),
    }
    poly3.save(MODELS_DIR)

    # ── 9. EXPONENTIAL DEGRADATION MODEL ─────────────────────────
    print("\n  ─── Exponential Degradation Model (physics-based) ───")
    exp_model = ExponentialDegradationModel(func='exponential')
    exp_model.fit(df)
    print("\n  Parameter Summary:")
    print(exp_model.get_summary().to_string(index=False))
    plot_degradation_fit(df, exp_model, 'Exponential Model')

    soh_true_all, soh_pred_all = [], []
    rul_true_all, rul_pred_all = [], []
    for bid in sorted(df['battery_id'].unique()):
        if bid not in exp_model.battery_params:
            continue
        bdf    = df[df['battery_id'] == bid].sort_values('cycle')
        cycles = bdf['cycle'].values.astype(float)
        soh_true_all.extend(bdf['SOH'].values)
        soh_pred_all.extend(exp_model.predict_soh(bid, cycles))
        rul_true_all.extend(bdf['RUL'].values)
        rul_pred_all.extend(
            [exp_model.predict_rul(bid, int(c)) for c in cycles]
        )

    all_results['Exponential Model'] = {
        'SOH': compute_metrics(np.array(soh_true_all), np.array(soh_pred_all), 'SOH'),
        'RUL': compute_metrics(np.array(rul_true_all), np.array(rul_pred_all), 'RUL'),
    }

    # ── 10. RESULTS TABLE ─────────────────────────────────────────
    print("\n[Step 5/7]  Evaluation Results:")
    flat_metrics = []
    for model_name, targets in all_results.items():
        for _, metrics in targets.items():
            flat_metrics.append({'Model': model_name, **metrics})

    print_metrics_table(flat_metrics)
    comparison_df = compare_models(all_results)

    # ── 11. COMPARISON PLOTS ──────────────────────────────────────
    print("[Step 6/7]  Generating comparison plots …")
    plot_model_comparison(comparison_df, metric='RMSE')
    plot_model_comparison(comparison_df, metric='R2')
    plot_model_comparison(comparison_df, metric='MAE')

    # ── 12. SAVE RESULTS ──────────────────────────────────────────
    print("[Step 7/7]  Saving results …")
    csv_path = os.path.join(RESULTS_DIR, 'analytical_model_results.csv')
    comparison_df.to_csv(csv_path, index=False)
    print(f"  💾 Results CSV  → {csv_path}")

    # ── FINAL SUMMARY ─────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  ✅  Analytical model pipeline complete!")
    print(f"  📁  Plots  → {RESULTS_DIR}")
    print(f"  📁  Models → {MODELS_DIR}")
    print("═" * 65 + "\n")


if __name__ == '__main__':
    main()
