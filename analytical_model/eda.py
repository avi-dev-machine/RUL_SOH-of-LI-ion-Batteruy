"""
Exploratory Data Analysis for Battery Dataset
===============================================
Generates plots for degradation curves, feature distributions,
correlation matrix, and capacity fade analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Colour palette
PALETTE = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4', '#FFEAA7',
           '#DDA0DD', '#98D8C8', '#F7DC6F', '#AED6F1']

plt.style.use('seaborn-v0_8-darkgrid')


def _save(fig: plt.Figure, fname: str) -> None:
    path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  📈 Saved: {path}")


def plot_degradation_curves(df: pd.DataFrame) -> None:
    """SOH and RUL vs cycle for every battery."""
    batteries = df['battery_id'].unique()
    colors    = PALETTE[:len(batteries)]

    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    fig.suptitle('Battery Degradation Curves', fontsize=16, fontweight='bold', y=1.01)

    for ax, target, ylabel in zip(
        axes,
        ['SOH', 'RUL'],
        ['State of Health (%)', 'Remaining Useful Life (cycles)']
    ):
        for bid, color in zip(batteries, colors):
            bdf = df[df['battery_id'] == bid].sort_values('cycle')
            ax.plot(bdf['cycle'], bdf[target], label=bid, color=color,
                    linewidth=2, alpha=0.85)

        if target == 'SOH':
            ax.axhline(y=80, color='#E74C3C', linestyle='--', linewidth=1.5,
                       alpha=0.7, label='EOL Threshold (80%)')

        ax.set_xlabel('Cycle Number', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{target} Degradation vs Cycle', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    _save(fig, 'eda_degradation_curves.png')


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """Histogram grid for all measurement features."""
    feature_cols = ['chI', 'chV', 'chT', 'disI', 'disV', 'disT', 'BCt', 'SOH', 'RUL']

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        ax.hist(df[col], bins=35, color=PALETTE[i % len(PALETTE)],
                edgecolor='white', alpha=0.85)
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.axvline(df[col].mean(), color='#E74C3C', linestyle='--',
                   linewidth=1.5, label=f'μ={df[col].mean():.2f}')
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    _save(fig, 'eda_feature_distributions.png')


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Heatmap of feature/target correlations."""
    cols   = ['chI', 'chV', 'chT', 'disI', 'disV', 'disT', 'BCt', 'cycle', 'SOH', 'RUL']
    corr   = df[cols].corr()
    mask   = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                ax=ax, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    _save(fig, 'eda_correlation_matrix.png')


def plot_capacity_fade(df: pd.DataFrame) -> None:
    """Scatter: BCt (capacity) vs cycle for each battery."""
    batteries = df['battery_id'].unique()
    colors    = PALETTE[:len(batteries)]

    fig, ax = plt.subplots(figsize=(12, 6))
    for bid, color in zip(batteries, colors):
        bdf = df[df['battery_id'] == bid].sort_values('cycle')
        ax.scatter(bdf['cycle'], bdf['BCt'], label=bid,
                   color=color, s=12, alpha=0.65)
        # Trend line
        z = np.polyfit(bdf['cycle'], bdf['BCt'], 1)
        p = np.poly1d(z)
        ax.plot(bdf['cycle'], p(bdf['cycle']), color=color,
                linewidth=2, linestyle='--', alpha=0.6)

    ax.set_xlabel('Cycle Number', fontsize=12)
    ax.set_ylabel('Battery Capacity  BCt (Ah)', fontsize=12)
    ax.set_title('Capacity Fade Over Charge/Discharge Cycles', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, 'eda_capacity_fade.png')


def plot_soh_rul_relationship(df: pd.DataFrame) -> None:
    """SOH vs RUL coloured by cycle."""
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(df['SOH'], df['RUL'], c=df['cycle'],
                    cmap='viridis', s=12, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Cycle Number')
    ax.set_xlabel('State of Health (%)', fontsize=12)
    ax.set_ylabel('Remaining Useful Life (cycles)', fontsize=12)
    ax.set_title('SOH vs RUL Relationship', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, 'eda_soh_vs_rul.png')


def plot_per_battery_boxplots(df: pd.DataFrame) -> None:
    """Box plots of SOH per battery."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Per-Battery Distributions', fontsize=14, fontweight='bold')

    for ax, target in zip(axes, ['SOH', 'RUL']):
        batteries = df['battery_id'].unique()
        data_list = [df.loc[df['battery_id'] == bid, target].values for bid in batteries]
        bp = ax.boxplot(data_list, patch_artist=True, labels=batteries)
        for patch, color in zip(bp['boxes'], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_xlabel('Battery ID', fontsize=11)
        ax.set_ylabel(target, fontsize=11)
        ax.set_title(f'{target} Distribution per Battery', fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    _save(fig, 'eda_per_battery_boxplots.png')


def run_eda(df: pd.DataFrame) -> None:
    """Run the complete EDA pipeline."""
    print("\n📊 Running Exploratory Data Analysis...")
    plot_degradation_curves(df)
    plot_feature_distributions(df)
    plot_correlation_matrix(df)
    plot_capacity_fade(df)
    plot_soh_rul_relationship(df)
    plot_per_battery_boxplots(df)
    print(f"✅ EDA complete. All plots saved in: {RESULTS_DIR}\n")
