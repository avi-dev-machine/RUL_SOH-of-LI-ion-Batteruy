"""
Monitoring Dashboard
=====================
Implements the final "Monitoring Dashboard" block from the Hybrid System
Architecture (Fig 12.1) and the "Dashboard & Reporting" block from the
Validation Architecture (Fig 11.1).

Dashboard panels (6-panel layout):
  [1] Battery SOH Degradation: Actual vs Physics vs Hybrid ANN
  [2] SOH Prediction — Actual vs Predicted scatter
  [3] RUL Prediction — Actual vs Predicted scatter
  [4] Training Loss (total, data, physics, monotonicity components)
  [5] Physics Validation Scores (PAS & DTCS per battery as gauge bars)
  [6] Model Comparison — Hybrid vs Standalone ANN (bar chart)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.30,
    'figure.dpi': 120,
})

# ── Colour palette ─────────────────────────────────────────────────────────────
C = {
    'actual'  : '#2ECC71',   # green  — ground truth
    'physics' : '#E67E22',   # orange — physics model
    'hybrid'  : '#3498DB',   # blue   — hybrid ANN
    'ann'     : '#9B59B6',   # purple — standalone ANN
    'eol'     : '#E74C3C',   # red    — End-of-Life line
    'pas'     : '#1ABC9C',   # teal   — PAS
    'dtcs'    : '#F39C12',   # amber  — DTCS
    'bg_dark' : '#1C2833',
    'text'    : '#EAECEE',
    'loss_t'  : '#3498DB',
    'loss_d'  : '#2ECC71',
    'loss_p'  : '#E67E22',
    'loss_m'  : '#E74C3C',
}


def _save(fig: plt.Figure, fname: str) -> None:
    path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Dashboard] Saved: {path}")


# ── Panel helpers ──────────────────────────────────────────────────────────────

def _degradation_panel(axes, df: pd.DataFrame,
                         soh_phys_col: str = 'SOH_phys') -> None:
    """Panel 1: Per-battery degradation — Actual / Physics / Hybrid."""
    batteries = sorted(df['battery_id'].unique())
    for ax, bid in zip(axes, batteries):
        bdf = df[df['battery_id'] == bid].sort_values('cycle')

        ax.scatter(bdf['cycle'], bdf['SOH'],
                   color=C['actual'], s=14, alpha=0.75,
                   label='Actual', zorder=5)

        if soh_phys_col in bdf.columns:
            ax.plot(bdf['cycle'], bdf[soh_phys_col],
                    color=C['physics'], linewidth=2.0,
                    linestyle='--', label='Physics', alpha=0.85)

        if 'SOH_hybrid' in bdf.columns:
            ax.plot(bdf['cycle'], bdf['SOH_hybrid'],
                    color=C['hybrid'], linewidth=2.5,
                    label='Hybrid ANN', alpha=0.95)

        ax.axhline(80, color=C['eol'], linestyle=':', linewidth=1.4,
                   alpha=0.8, label='EOL (80%)')
        ax.set_xlabel('Cycle', fontsize=10)
        ax.set_ylabel('SOH (%)', fontsize=10)
        ax.set_title(f'Battery {bid}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')


def _scatter_panel(ax, y_true: np.ndarray, y_pred: np.ndarray,
                    target: str, color: str) -> None:
    """Actual vs Predicted scatter with perfect-fit line."""
    lo = min(y_true.min(), y_pred.min()) * 0.98
    hi = max(y_true.max(), y_pred.max()) * 1.02
    ax.scatter(y_true, y_pred, color=color, s=14, alpha=0.45, edgecolors='none')
    ax.plot([lo, hi], [lo, hi], '--', color='#2C3E50', linewidth=1.8, label='Ideal')
    ax.set_xlabel(f'Actual {target}', fontsize=10)
    ax.set_ylabel(f'Predicted {target}', fontsize=10)
    ax.set_title(f'{target}: Actual vs Hybrid Predicted',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)


def _loss_panel(ax, history: dict) -> None:
    """Training & validation loss decomposition."""
    ax.plot(history['train_loss'], color=C['loss_t'], linewidth=2.0, label='Total (train)')
    ax.plot(history['val_loss'],   color=C['loss_t'], linewidth=1.5,
            linestyle='--', label='Total (val)', alpha=0.75)
    if 'train_data' in history:
        ax.plot(history['train_data'], color=C['loss_d'], linewidth=1.5,
                linestyle='-', label='Data loss', alpha=0.85)
    if 'train_phys' in history:
        ax.plot(history['train_phys'], color=C['loss_p'], linewidth=1.5,
                linestyle='-', label='Phys loss', alpha=0.85)
    if 'train_mono' in history:
        ax.plot(history['train_mono'], color=C['loss_m'], linewidth=1.5,
                linestyle='-', label='Mono loss', alpha=0.85)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss (MSE)', fontsize=10)
    ax.set_title('Physics-Informed Training Loss', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_yscale('log')


def _gauge_panel(ax, pas: float, dtcs: float) -> None:
    """Horizontal gauge bars for PAS and DTCS."""
    metrics  = {'PAS\n(Physics Adherence Score)': (pas, C['pas']),
                 'DTCS\n(Trajectory Consistency)': (dtcs, C['dtcs'])}
    ys       = list(range(len(metrics)))[::-1]
    for y_pos, (label, (val, color)) in zip(ys, metrics.items()):
        ax.barh(y_pos, 1.0, height=0.55, color='#D5DBDB', alpha=0.5)
        ax.barh(y_pos, val, height=0.55, color=color, alpha=0.90)
        ax.text(val + 0.01, y_pos, f'{val:.3f}', va='center',
                fontsize=13, fontweight='bold', color=color)
        ax.text(-0.02, y_pos, label, va='center', ha='right',
                fontsize=9, color='#2C3E50')
    ax.set_xlim(-0.25, 1.15)
    ax.set_yticks([])
    ax.set_xlabel('Score', fontsize=10)
    ax.set_title('Physics Validation Scores', fontsize=11, fontweight='bold')
    ax.axvline(1.0, color='black', linewidth=0.8, linestyle=':')
    ax.grid(False, axis='y')


def _comparison_panel(ax, comparison_data: dict) -> None:
    """Bar chart comparing hybrid vs standalone ANN metrics."""
    models  = list(comparison_data.keys())
    metrics = ['RMSE_SOH', 'MAE_SOH', 'RMSE_RUL', 'MAE_RUL']
    labels  = ['SOH RMSE', 'SOH MAE', 'RUL RMSE', 'RUL MAE']
    colors  = [C['hybrid'], C['ann']]

    x      = np.arange(len(metrics))
    width  = 0.35
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [comparison_data[model].get(m, 0) for m in metrics]
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      color=color, alpha=0.85, label=model, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Error', fontsize=10)
    ax.set_title('Hybrid vs Standalone ANN', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)


# ── Main dashboard ─────────────────────────────────────────────────────────────

def create_monitoring_dashboard(df_full:    pd.DataFrame,
                                  y_true:     np.ndarray,
                                  y_pred:     np.ndarray,
                                  history:    dict,
                                  pas:        float,
                                  dtcs:       float,
                                  metrics_df: pd.DataFrame,
                                  comparison: dict = None) -> None:
    """
    Render the full Monitoring Dashboard (Fig 12.1 final block).

    Args:
        df_full    : Full DataFrame with battery_id, cycle, SOH,
                     SOH_phys, SOH_hybrid columns
        y_true     : (N_test, 2) test ground truth [SOH, RUL]
        y_pred     : (N_test, 2) hybrid ANN predictions [SOH_pred, RUL_pred]
        history    : Training loss history dict
        pas        : Overall PAS score
        dtcs       : Overall DTCS score
        metrics_df : DataFrame from performance_evaluation()
        comparison : {'Hybrid ANN': {...}, 'Standalone ANN': {...}}
    """
    batteries = sorted(df_full['battery_id'].unique())
    n_batt    = len(batteries)

    fig = plt.figure(figsize=(22, 28), facecolor='white')
    fig.suptitle(
        'Hybrid Physics-Informed ANN — Monitoring Dashboard\n'
        'Battery SOH & RUL Prediction',
        fontsize=18, fontweight='bold', color='#1C2833', y=0.98
    )

    gs = gridspec.GridSpec(4, 3, figure=fig,
                            hspace=0.48, wspace=0.38,
                            top=0.94, bottom=0.04,
                            left=0.06, right=0.97)

    # ── Row 0: Battery degradation curves (spans all 3 cols) ──────
    degrad_axes = []
    for col in range(n_batt):
        degrad_axes.append(fig.add_subplot(gs[0, col]))
    _degradation_panel(degrad_axes, df_full)

    # ── Row 1: SOH scatter | RUL scatter | Training loss ──────────
    ax_soh = fig.add_subplot(gs[1, 0])
    ax_rul = fig.add_subplot(gs[1, 1])
    ax_his = fig.add_subplot(gs[1, 2])
    _scatter_panel(ax_soh, y_true[:, 0], y_pred[:, 0], 'SOH (%)', C['hybrid'])
    _scatter_panel(ax_rul, y_true[:, 1], y_pred[:, 1], 'RUL (cycles)', C['ann'])
    _loss_panel(ax_his, history)

    # ── Row 2: Residual dist SOH | Residual dist RUL | PAS/DTCS gauges ─
    ax_res_soh = fig.add_subplot(gs[2, 0])
    ax_res_rul = fig.add_subplot(gs[2, 1])
    ax_gauge   = fig.add_subplot(gs[2, 2])

    res_soh = y_pred[:, 0] - y_true[:, 0]
    res_rul = y_pred[:, 1] - y_true[:, 1]
    for ax, res, target, color in [
        (ax_res_soh, res_soh, 'SOH (%)', C['hybrid']),
        (ax_res_rul, res_rul, 'RUL (cycles)', C['ann']),
    ]:
        ax.hist(res, bins=35, color=color, edgecolor='white', alpha=0.85)
        ax.axvline(0, color='#2C3E50', linewidth=2, linestyle='--')
        ax.axvline(res.mean(), color=C['eol'], linewidth=1.5,
                   label=f'Mean: {res.mean():.2f}')
        ax.set_xlabel(f'Residual {target}', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{target} Residual Distribution', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)

    _gauge_panel(ax_gauge, pas, dtcs)

    # ── Row 3: Metrics table | Model comparison ────────────────────
    ax_table = fig.add_subplot(gs[3, 0:2])
    ax_comp  = fig.add_subplot(gs[3, 2])

    ax_table.axis('off')
    col_labels = ['Target', 'MAE', 'RMSE', 'R²', 'MAPE(%)', 'Max Error', 'PAS', 'DTCS']
    table_data = []
    for _, row in metrics_df.iterrows():
        table_data.append([
            row.get('Target', ''),
            f"{row.get('MAE',0):.4f}",
            f"{row.get('RMSE',0):.4f}",
            f"{row.get('R2',0):.4f}",
            f"{row.get('MAPE(%)',0):.4f}",
            f"{row.get('Max Error',0):.4f}",
            f"{row.get('PAS',0):.4f}",
            f"{row.get('DTCS',0):.4f}",
        ])

    tbl = ax_table.table(cellText=table_data, colLabels=col_labels,
                          loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#EBF5FB')
        cell.set_edgecolor('#BDC3C7')
    ax_table.set_title('Error Calculation & Metrics Layer',
                        fontsize=12, fontweight='bold', pad=12)

    if comparison:
        _comparison_panel(ax_comp, comparison)

    _save(fig, 'hybrid_monitoring_dashboard.png')


def plot_per_battery_validation(df_full: pd.DataFrame,
                                  pas_per_battery:  dict,
                                  dtcs_per_battery: dict) -> None:
    """Per-battery PAS and DTCS breakdown bar chart."""
    batteries = sorted(df_full['battery_id'].unique())
    n = len(batteries)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Per-Battery Physics Validation Scores',
                 fontsize=14, fontweight='bold')

    for ax, metric, data, color in [
        (axes[0], 'PAS',  pas_per_battery,  C['pas']),
        (axes[1], 'DTCS', dtcs_per_battery, C['dtcs']),
    ]:
        bids  = [b for b in batteries if b in data]
        vals  = [data[b] for b in bids]
        bars  = ax.bar(bids, vals, color=color, alpha=0.85, edgecolor='white', width=0.5)
        ax.set_ylim(0, 1.15)
        ax.axhline(1.0, color='#2C3E50', linewidth=1, linestyle=':')
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} per Battery', fontsize=12, fontweight='bold')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.02, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold',
                    color=color)

    plt.tight_layout()
    _save(fig, 'hybrid_per_battery_validation.png')
