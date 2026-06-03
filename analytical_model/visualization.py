"""
Visualization Utilities for Analytical Model
=============================================
Provides prediction plots, residual plots, feature importance charts,
degradation curve overlays, and model comparison bar charts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')

COLORS = {
    'actual'  : '#4ECDC4',
    'pred'    : '#FF6B6B',
    'perfect' : '#2C3E50',
    'residual': '#45B7D1',
    'bar'     : ['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'],
}


def _save(fig: plt.Figure, fname: str) -> None:
    path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    📈 Saved: {path}")


def plot_predictions_vs_actual(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                target: str,
                                model_name: str) -> None:
    """Scatter (actual vs predicted) + residual histogram side by side."""
    y_true    = np.asarray(y_true).flatten()
    y_pred    = np.asarray(y_pred).flatten()
    residuals = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{model_name}  —  {target} Predictions', fontsize=14, fontweight='bold')

    # ── Actual vs Predicted ────────────────────────────────────────
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.45, color=COLORS['actual'],
               edgecolors='white', s=18, label='Samples')
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], '--', color=COLORS['perfect'],
            linewidth=2, label='Ideal (y = ŷ)')
    ax.set_xlabel(f'Actual {target}', fontsize=11)
    ax.set_ylabel(f'Predicted {target}', fontsize=11)
    ax.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

    # ── Residuals ──────────────────────────────────────────────────
    ax = axes[1]
    ax.hist(residuals, bins=35, color=COLORS['residual'],
            edgecolor='white', alpha=0.85)
    ax.axvline(0, color=COLORS['perfect'], linestyle='--', linewidth=2)
    ax.axvline(residuals.mean(), color=COLORS['pred'], linestyle='-',
               linewidth=1.5, label=f'Mean={residuals.mean():.3f}')
    ax.set_xlabel('Residual  (ŷ − y)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    safe = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    _save(fig, f'{safe}_{target.lower()}_predictions.png')


def plot_degradation_fit(df: pd.DataFrame,
                          exp_model,
                          model_name: str = 'Exponential Model') -> None:
    """Overlay fitted degradation curves on actual SOH scatter."""
    batteries = sorted(df['battery_id'].unique())
    n   = len(batteries)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    fig.suptitle(f'{model_name} — SOH Degradation Fit', fontsize=14, fontweight='bold')

    for ax, bid in zip(axes[0], batteries):
        bdf    = df[df['battery_id'] == bid].sort_values('cycle')
        cycles = bdf['cycle'].values.astype(float)
        soh    = bdf['SOH'].values

        ax.scatter(cycles, soh, color=COLORS['actual'], s=12,
                   alpha=0.7, label='Actual SOH', zorder=5)

        if hasattr(exp_model, 'predict_soh') and bid in exp_model.battery_params:
            soh_fit = exp_model.predict_soh(bid, cycles)
            ax.plot(cycles, soh_fit, color=COLORS['pred'],
                    linewidth=2.5, label='Fitted Curve')

        ax.axhline(y=80, color='gray', linestyle='--',
                   linewidth=1.2, alpha=0.6, label='EOL (80%)')
        ax.set_xlabel('Cycle', fontsize=10)
        ax.set_ylabel('SOH (%)', fontsize=10)
        ax.set_title(f'Battery  {bid}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    safe = model_name.lower().replace(' ', '_')
    _save(fig, f'{safe}_degradation_fit.png')


def plot_feature_importance(coeff_dict: dict, model_name: str) -> None:
    """Horizontal bar chart of feature coefficients per target."""
    for target, coeffs in coeff_dict.items():
        features = list(coeffs.keys())
        values   = list(coeffs.values())
        bar_colors = [COLORS['actual'] if v >= 0 else COLORS['pred'] for v in values]

        # Sort by absolute magnitude
        order    = np.argsort(np.abs(values))
        features = [features[i] for i in order]
        values   = [values[i]   for i in order]
        bar_colors = [bar_colors[i] for i in order]

        fig, ax = plt.subplots(figsize=(10, max(5, len(features) * 0.4 + 1)))
        ax.barh(features, values, color=bar_colors, edgecolor='white', alpha=0.85)
        ax.axvline(0, color=COLORS['perfect'], linewidth=0.8)
        ax.set_xlabel('Coefficient Value', fontsize=11)
        ax.set_title(f'{model_name} — Feature Importance ({target})',
                     fontsize=12, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        safe = model_name.lower().replace(' ', '_')
        _save(fig, f'{safe}_coefficients_{target.lower()}.png')


def plot_model_comparison(comparison_df: pd.DataFrame,
                            metric: str = 'RMSE') -> None:
    """Grouped bar chart comparing all models on SOH and RUL."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Model Comparison  —  {metric}', fontsize=15, fontweight='bold')

    for ax, target in zip(axes, ['SOH', 'RUL']):
        sub   = comparison_df[comparison_df['Target'] == target].reset_index(drop=True)
        n     = len(sub)
        colors = COLORS['bar'][:n]

        bars = ax.bar(range(n), sub[metric], color=colors, edgecolor='white', width=0.6)
        ax.set_xticks(range(n))
        ax.set_xticklabels(sub['Model'], rotation=20, ha='right', fontsize=9)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{target} Prediction', fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.35)
        ax.spines[['top', 'right']].set_visible(False)

        for bar, val in zip(bars, sub[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    _save(fig, f'model_comparison_{metric.lower()}.png')
