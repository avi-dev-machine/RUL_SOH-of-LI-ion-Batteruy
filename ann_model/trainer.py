"""
Training Visualizations for ANN Battery Model (PyTorch)
========================================================
History plots, prediction scatter, residual distributions, timeline overlay.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')

PALETTE = {
    'train'   : '#4ECDC4',
    'val'     : '#FF6B6B',
    'soh'     : '#4ECDC4',
    'rul'     : '#45B7D1',
    'perfect' : '#2C3E50',
}


def _save(fig: plt.Figure, fname: str) -> None:
    path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  📈 Saved: {path}")


def plot_training_history(history: dict, arch: str = '') -> None:
    """
    Plot loss and MAE curves.

    Args:
        history : dict with keys 'train_loss', 'val_loss', 'train_mae', 'val_mae'
        arch    : architecture label for the filename
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'ANN Training History  [{arch}]' if arch else 'ANN Training History',
                 fontsize=15, fontweight='bold')

    # ── Loss ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(history['train_loss'], label='Train Loss', color=PALETTE['train'], linewidth=2)
    ax.plot(history['val_loss'],   label='Val Loss',   color=PALETTE['val'],   linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MSE Loss', fontsize=11)
    ax.set_title('Loss', fontsize=12, fontweight='bold')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)

    # ── MAE ───────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(history['train_mae'], label='Train MAE', color=PALETTE['train'], linewidth=2)
    ax.plot(history['val_mae'],   label='Val MAE',   color=PALETTE['val'],   linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MAE', fontsize=11)
    ax.set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    tag = f'_{arch}' if arch else ''
    _save(fig, f'ann{tag}_training_history.png')


def plot_predictions(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      target_names: list = None,
                      arch: str = '') -> None:
    """
    Scatter (actual vs predicted) + residual histogram.

    Args:
        y_true       : (N, 2)
        y_pred       : (N, 2)
        target_names : ['SOH', 'RUL']
        arch         : architecture tag for filename
    """
    target_names = target_names or ['SOH', 'RUL']
    colors = [PALETTE['soh'], PALETTE['rul']]
    n = len(target_names)

    fig, axes = plt.subplots(2, n, figsize=(7 * n, 12))
    fig.suptitle(f'ANN Predictions vs Actual  [{arch}]' if arch else 'ANN Predictions vs Actual',
                 fontsize=16, fontweight='bold')

    for i, (target, color) in enumerate(zip(target_names, colors)):
        true_v = y_true[:, i]
        pred_v = y_pred[:, i]
        resid  = pred_v - true_v

        ax = axes[0, i]
        ax.scatter(true_v, pred_v, alpha=0.35, color=color, s=14, edgecolors='white')
        lo = min(true_v.min(), pred_v.min())
        hi = max(true_v.max(), pred_v.max())
        ax.plot([lo, hi], [lo, hi], '--', color=PALETTE['perfect'],
                linewidth=2, label='Ideal')
        ax.set_xlabel(f'Actual {target}', fontsize=11)
        ax.set_ylabel(f'Predicted {target}', fontsize=11)
        ax.set_title(f'{target}: Actual vs Predicted', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

        ax = axes[1, i]
        ax.hist(resid, bins=40, color=color, edgecolor='white', alpha=0.85)
        ax.axvline(0, color=PALETTE['perfect'], linestyle='--', linewidth=2)
        ax.axvline(resid.mean(), color='red', linestyle='-', linewidth=1.5,
                   label=f'Mean: {resid.mean():.3f}')
        ax.set_xlabel('Residual  (ŷ − y)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{target}: Residual Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    tag = f'_{arch}' if arch else ''
    _save(fig, f'ann{tag}_predictions.png')


def plot_timeline(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   target_names: list = None,
                   n_samples:    int  = 120,
                   arch:         str  = '') -> None:
    """Timeline overlay of actual vs predicted for the first n_samples."""
    target_names = target_names or ['SOH', 'RUL']
    colors       = [PALETTE['soh'], PALETTE['rul']]

    n   = min(n_samples, len(y_true))
    idx = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Prediction Timeline  [{arch}]' if arch else 'Prediction Timeline',
                 fontsize=14, fontweight='bold')

    for i, (ax, target, color) in enumerate(zip(axes, target_names, colors)):
        ax.plot(idx, y_true[:n, i], color=color,      linewidth=2, label='Actual',    alpha=0.85)
        ax.plot(idx, y_pred[:n, i], color=PALETTE['val'], linewidth=2,
                linestyle='--', label='Predicted', alpha=0.90)
        ax.fill_between(idx, y_true[:n, i], y_pred[:n, i],
                        alpha=0.15, color='gray', label='Error band')
        ax.set_xlabel('Sample Index', fontsize=11)
        ax.set_ylabel(target, fontsize=11)
        ax.set_title(f'{target} — Prediction Timeline', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    tag = f'_{arch}' if arch else ''
    _save(fig, f'ann{tag}_timeline.png')


def plot_error_vs_cycle(y_true:  np.ndarray,
                         y_pred:  np.ndarray,
                         cycles:  np.ndarray,
                         target:  str = 'SOH',
                         col_idx: int = 0,
                         arch:    str = '') -> None:
    """Scatter: prediction error vs cycle number."""
    err = np.abs(y_pred[:, col_idx] - y_true[:, col_idx])
    n   = min(len(err), len(cycles))

    fig, ax = plt.subplots(figsize=(12, 5))
    sc = ax.scatter(cycles[:n], err[:n], c=cycles[:n], cmap='viridis', s=12, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Cycle Number')
    ax.set_xlabel('Cycle Number', fontsize=11)
    ax.set_ylabel(f'|{target} Error|', fontsize=11)
    ax.set_title(f'{target} Absolute Error vs Cycle [{arch}]', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.35)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    tag = f'_{arch}' if arch else ''
    _save(fig, f'ann{tag}_error_vs_cycle_{target.lower()}.png')
