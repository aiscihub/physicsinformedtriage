#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from pipeline.training.v1.generate_figure_both_cv_ligand import load_results_from_pkl, generate_comparison_figure, \
    plot_timescale_horizon, plot_triage_efficiency, plot_early_time_separation_compact

# Style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",   # Windows / macOS
        "Times",             # macOS fallback
        "Nimbus Roman",      # Linux
        "DejaVu Serif"       # Matplotlib default (always exists)
    ],
    "font.size": 12,                 # base font size
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 18,
})

def load_data(csv_path):
    """Load screening report data."""
    df = pd.read_csv(csv_path)
    return df


def plot_cost_savings_vs_timescale(df, ax):
    """Plot computational savings vs simulation timescale using best-performing model."""

    colors = {'protein_logo': '#4C72B0', 'pocket_gkf': '#55A868'}
    labels = {'protein_logo': 'Cross-protein CV', 'pocket_gkf': 'Cross-pocket CV'}

    for strategy in ['protein_logo', 'pocket_gkf']:
        times = sorted(df['timescale_ns'].unique())

        savings_20 = []
        savings_10 = []

        for t in times:
            best20 = select_best_model(df, strategy, t, fpr=0.20)
            best10 = select_best_model(df, strategy, t, fpr=0.10)

            savings_20.append(
                best20['cost_saved@fpr0.20'] * 100 if best20 is not None else np.nan
            )
            savings_10.append(
                best10['cost_saved@fpr0.10'] * 100 if best10 is not None else np.nan
            )

        # Plot once per strategy (outside the inner loop)
        ax.plot(
            times, savings_20, 'o-',
            color=colors[strategy], linewidth=2, markersize=8,
            label=f'{labels[strategy]} (best, FPR≤20%)'
        )
        ax.plot(
            times, savings_10, 's--',
            color=colors[strategy], linewidth=1.5, markersize=6, alpha=0.7,
            label=f'{labels[strategy]} (best, FPR≤10%)'
        )

    ax.set_xlabel('Early MD Simulation Time (ns)')
    ax.set_ylabel('Computational Cost Saved (%)')
    #ax.set_title('B. Cost Savings vs Simulation Length', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim([0, 45])
    ax.set_xlim([0, 15])
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)

    # # Highlight optimal region
    # ax.axvspan(2, 5, color='yellow', alpha=0.15)
    # ax.text(3.5, 42, 'Sweet spot', fontsize=9,
    #         ha='center', style='italic', color='#666')

def select_best_model(df, strategy, timescale, fpr=0.20):
    """
    Select the best model for a given strategy and timescale
    that ACTUALLY satisfies the FPR constraint.
    """
    fpr_col = f'fpr{fpr:.2f}'
    metric_suffix = f'fpr{fpr:.2f}'

    actual_fpr_col = f'actual_fpr@thr({metric_suffix})'


    subset = df[
        (df['strategy'] == strategy) &
        (df['timescale_ns'] == timescale) &
        (df[actual_fpr_col] <= fpr + 1e-9)
        ].copy()

    if subset.empty:
        return None

    subset['recall'] = subset[f'recall@{metric_suffix}']
    subset['cost'] = subset[f'cost_saved@{metric_suffix}']

    return subset.sort_values(
        by=['recall', 'cost'],
        ascending=False
    ).iloc[0]

def save_panel_b_cost_savings(df, output_path):
    """
    Save Panel B (Cost Savings vs Simulation Length)
    as a standalone, publication-ready figure.
    """
    plt.figure(figsize=(6.5, 4.8))

    ax = plt.gca()
    plot_cost_savings_vs_timescale(df, ax)

    plt.title(
        '',
        fontsize=13,
        fontweight='bold',
        pad=10
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved Panel B: {output_path}")


def main():
    date_tag = "20251225"
    csv_path = Path(f"dataset/screening_report_both_cv_{date_tag}.csv")
    
    print("="*60)
    print("GENERATING OPERATIONAL TRIAGE VISUALIZATIONS")
    print("="*60)
    
    # Load data
    df = load_data(csv_path)
    print(f"Loaded {len(df)} rows from screening report")

    # ---- Save standalone Panel B ----
    panel_b_path = Path(f"./dataset/panel_b_cost_savings_{date_tag}.png")
    save_panel_b_cost_savings(df, panel_b_path)

    print("\n" + "="*60)
    print("KEY OPERATING POINTS (BEST MODELS)")
    print("="*60)


    results_dict = load_results_from_pkl(Path(f'./dataset/results_both_cv_{date_tag}.pkl'))
    generate_comparison_figure(results_dict, Path(f"./dataset/comprehensive_ml_both_cv_{date_tag}.png"),
                               Path(f"dataset/fingerprint_summary_with_components_even_ac_4d_drift_{date_tag}.csv"))
    plot_timescale_horizon(
        results_dict,
        output=f"./dataset/figure1_timescale_horizon_{date_tag}.png"
    )


    plot_triage_efficiency(
        results_dict,
        output=f"./dataset/figure2_triage_efficiency_{date_tag}.png"
    )

    plot_early_time_separation_compact(
        features_path=Path(f"dataset/fingerprint_summary_with_components_even_ac_4d_drift_{date_tag}.csv"),
        timescale=2,
        output=f"./dataset/early_time_separation_2ns_{date_tag}.png"
    )
if __name__ == "__main__":
    main()
