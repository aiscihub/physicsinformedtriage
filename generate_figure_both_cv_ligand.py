#!/usr/bin/env python3
"""
Generate Comprehensive ML Analysis Figure

This script trains ML models and generates the comprehensive 6-panel figure
showing model comparison, ROC curves, PR curves, confusion matrix, and key findings.

This is the SAME figure as ml_comprehensive_analysis.png - the one you liked!

Usage:
    python generate_comprehensive_figure.py \
        --features fingerprint_summary_with_components.csv \
        --output comprehensive_ml_results.png
"""

import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix
)
import warnings

from pipeline.training.generate_table_iii_median_iqr import generate_table_iii

warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
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

RESULTS = []

TIMESCALES = [2, 4, 6, 8, 10, 12, 14]
TIMESCALES = [1,2, 3, 4, 5, 6, 7, 8, 10, 12, 14]
CV_SETTINGS = {
    "protein": {
        "group_by": "protein",
        "cv_mode": "logo",
        "label": "Cross-protein (LOPO)"
    },
    "pocket": {
        "group_by": "pocket",
        "cv_mode": "gkf",
        "label": "Cross-pocket (protein|pocket)"
    }
}


FEATURE_COLS = [
    'slope',
    'rmsd_var',
    'mean_disp',
    'var_disp',
    # 'f_simple',
    # 'f_hbond',
    # 'f_residue'
]

from sklearn.base import clone

def select_best_time_and_model(results_by_timescale, metric="auc"):
    """
    Return (best_time, best_model_name, best_score) across all timescales and models.
    """
    best_time = None
    best_model = None
    best_score = -np.inf

    for t, models in results_by_timescale.items():
        for model_name, r in models.items():
            score = r.get(metric, None)
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best_time = t
                best_model = model_name

    return best_time, best_model, best_score


def load_data(features_path: Path, timescale: float, group_by='protein'):
    """
    Load and prepare data.

    Args:
        features_path: Path to CSV
        timescale: Timescale to extract
        group_by: 'protein' (leave-one-protein-out),
                  'pocket' (pocket-level CV with protein|pocket groups), or
                  'ligand_pocket' (protein|ligand|pocket groups - if ligand_name exists)
    """
    df = pd.read_csv(features_path)

    # Check if ligand_name column exists
    has_ligand = 'ligand_name' in df.columns

    exclude = (
            ((df["protein"] == "MDR2_TRIRC") & (df["pocket"] == "pocket9")) |
            ((df["protein"] == "SNQ2_CANGA") & (df["pocket"] == "pocket2"))
    )

    df = df[~exclude].reset_index(drop=True)

    df = df[df['label_unstable'].notna()].copy()
    df_t = df[df['time_ns'] == timescale].copy()
    #
    # # --- log-transform heavy-tailed features ---
    # for col in ["slope", "var_disp"]:
    #     if col in df_t.columns:
    #         df_t[col] = np.log1p(df_t[col])

    X = df_t[FEATURE_COLS].values
    y = df_t['label_unstable'].values


    if group_by == 'ligand_pocket' and has_ligand:
        # Group by protein|ligand|pocket to keep replicas together
        groups = (df_t['protein'].astype(str) + '|' +
                  df_t['ligand_name'].astype(str) + '|' +
                  df_t['pocket'].astype(str)).values
    elif group_by == 'pocket':
        # Group by protein|pocket to keep replicas together
        groups = (df_t['protein'].astype(str) + '|' + df_t['pocket'].astype(str)).values
    else:
        # Original: group by protein only
        groups = df_t['protein'].values

    return X, y, groups


def get_models():
    """Return models to evaluate."""
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', C=3, random_state=42

        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_split=3,
            class_weight='balanced', random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03, random_state=42
        ),
        # 'Neural Network': MLPClassifier(
        #     hidden_layer_sizes=(32, 16), activation='relu', solver='adam',
        #     max_iter=500, random_state=42, early_stopping=True
        # )
    }


def train_model_cv(X, y, groups, model, cv_mode='logo', target_fpr=0.10,
                   fpr_targets=None, t_ns=None, long_ns=20.0):
    """
    Train model with group-aware cross-validation.

    Args:
        cv_mode: 'logo' (LeaveOneGroupOut) or 'gkf' (GroupKFold with 5 splits)
        fpr_targets: list of FPR targets for operating point evaluation (e.g., [0.10, 0.20])
        t_ns: current timescale in ns (needed for cost_saved calculation)
        long_ns: baseline simulation length for cost calculation

    Returns:
        y_true_all, y_prob_all, y_pred_all, fold_aucs, fold_pr_aucs, fold_ops
        where fold_ops is a list of per-fold operating point metrics
    """
    from sklearn.model_selection import GroupKFold

    if fpr_targets is None:
        fpr_targets = [0.10, 0.20]

    if cv_mode == 'gkf':
        cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    else:
        cv = LeaveOneGroupOut()

    scaler = StandardScaler()

    y_true_all = []
    y_prob_all = []
    y_pred_all = []
    fold_aucs = []
    fold_pr_aucs = []
    fold_ops = []  # NEW: per-fold operating point metrics

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        cloned_model = clone(model)
        cloned_model.fit(X_train_scaled, y_train)

        # Probabilities on train (for threshold selection) and test
        y_prob_train = cloned_model.predict_proba(X_train_scaled)[:, 1]
        y_prob_test = cloned_model.predict_proba(X_test_scaled)[:, 1]

        # Default threshold using target_fpr (backward compatible)
        thr, _, _ = _threshold_at_target_fpr(y_train, y_prob_train, target_fpr=target_fpr)
        y_pred_test = (y_prob_test >= thr).astype(int)

        y_true_all.extend(y_test)
        y_prob_all.extend(y_prob_test)
        y_pred_all.extend(y_pred_test)

        if len(np.unique(y_test)) > 1:
            fold_aucs.append(roc_auc_score(y_test, y_prob_test))
            fold_pr_aucs.append(average_precision_score(y_test, y_prob_test))

        # === NEW: Operating-point evaluation per fold ===
        for fpr_target in fpr_targets:
            # Select threshold on TRAIN set at this FPR target
            tau, _, _ = _threshold_at_target_fpr(y_train, y_prob_train, fpr_target)

            if not np.isfinite(tau):
                continue

            y_hat = (y_prob_test >= tau).astype(int)

            # Confusion matrix elements
            tp = ((y_hat == 1) & (y_test == 1)).sum()
            fn = ((y_hat == 0) & (y_test == 1)).sum()
            fp = ((y_hat == 1) & (y_test == 0)).sum()
            tn = ((y_hat == 0) & (y_test == 0)).sum()

            # Metrics
            recall = tp / (tp + fn + 1e-8)
            achieved_fpr = fp / (fp + tn + 1e-8)
            triage_rate = y_hat.mean()

            # Cost saved (only if t_ns is provided)
            cost_saved = np.nan
            if t_ns is not None:
                cost_saved = triage_rate * (long_ns - t_ns) / long_ns

            fold_ops.append({
                "fold": fold_id,
                "time_ns": t_ns,
                "fpr_target": fpr_target,
                "recall": float(recall),
                "achieved_fpr": float(achieved_fpr),
                "triage_rate": float(triage_rate),
                "cost_saved": float(cost_saved),
                "n_test": len(y_test),
                "tp": int(tp),
                "fn": int(fn),
                "fp": int(fp),
                "tn": int(tn),
            })

    return (np.array(y_true_all), np.array(y_prob_all), np.array(y_pred_all),
            np.array(fold_aucs), np.array(fold_pr_aucs), fold_ops)


def _log_cv_fold_summary(y, groups, cv, header: str = ""):
    """Log per-fold test-set sizes and class counts (useful for LOPO small-n interpretation)."""
    y = np.asarray(y).astype(int)
    uniq_groups = np.unique(groups)
    fold_sizes = []
    if header:
        log.info(header)
    for fold_id, (_, test_idx) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
        y_te = y[test_idx]
        n_te = len(test_idx)
        n_pos = int(y_te.sum())
        n_neg = int(n_te - n_pos)
        held_out = np.unique(groups[test_idx])
        held_out_str = held_out[0] if len(held_out) == 1 else f"{len(held_out)} groups"
        fold_sizes.append(n_te)
        log.info(
            f"  Fold {fold_id:2d}: n_test={n_te:2d} (unstable={n_pos:2d}, stable={n_neg:2d}), held_out={held_out_str}"
        )
    if fold_sizes:
        fold_sizes = np.asarray(fold_sizes)
        log.info(
            f"  Test-set size across folds: min={fold_sizes.min()}, median={int(np.median(fold_sizes))}, max={fold_sizes.max()} (n_folds={len(fold_sizes)}, n_groups={len(uniq_groups)})"
        )


def evaluate_all_models(
        features_path: Path,
        timescales: list,
        group_by: str = "protein",
        cv_mode: str = "logo",
        GOAL_FPR: float = 0.10,
        fpr_targets: tuple = (0.10, 0.20),
        long_ns: float = 20.0,
        print_fold_summary: bool = False,
        fold_summary_timescale: float = 2.0,
):
    """
    Train and evaluate all models at all timescales.

    Metrics are reported as mean ± std across CV folds.
    Now also collects per-fold operating point metrics for Table III.
    """

    log.info("=" * 80)
    log.info("TRAINING MODELS")
    log.info("=" * 80)
    log.info(f"Grouping strategy: {group_by}")
    log.info(f"CV mode: {cv_mode}")

    results_by_timescale = {}
    all_fold_ops = []  # NEW: collect all fold-level operating points
    models_dict = get_models()

    for t in timescales:
        log.info(f"\nTimescale: {t} ns")
        X, y, groups = load_data(features_path, t, group_by=group_by)

        log.info(
            f"  Samples: {len(X)}, "
            f"Unstable: {int(y.sum())} ({100 * y.mean():.1f}%)"
        )
        log.info(f"  Unique groups: {len(np.unique(groups))}")

        # Optional: print per-fold test-set sizes (helpful for LOPO small-n interpretation)
        if print_fold_summary and float(t) == float(fold_summary_timescale):
            from sklearn.model_selection import GroupKFold
            if cv_mode == 'gkf':
                cv_dbg = GroupKFold(n_splits=min(5, len(np.unique(groups))))
            else:
                cv_dbg = LeaveOneGroupOut()
            _log_cv_fold_summary(
                y=y,
                groups=groups,
                cv=cv_dbg,
                header=f"Per-fold test-set sizes at t={t} ns ({CV_SETTINGS.get('protein' if cv_mode=='logo' else 'pocket', {}).get('label', cv_mode)}; group_by={group_by})"
            )

        results_by_timescale[t] = {}

        for model_name, model in models_dict.items():
            try:
                (
                    y_true,
                    y_prob,
                    y_pred,
                    fold_aucs,
                    fold_pr_aucs,
                    fold_ops,  # NEW: per-fold operating point metrics
                ) = train_model_cv(
                    X,
                    y,
                    groups,
                    model,
                    cv_mode=cv_mode,
                    target_fpr=GOAL_FPR,
                    fpr_targets=list(fpr_targets),
                    t_ns=t,
                    long_ns=long_ns,
                )

                # Guard against degenerate cases
                if len(fold_aucs) == 0:
                    log.warning(
                        f"    {model_name:20s}: "
                        "Skipped (no valid CV folds with both classes)"
                    )
                    continue

                auc_mean = float(np.mean(fold_aucs))
                auc_std = float(np.std(fold_aucs))
                pr_mean = float(np.mean(fold_pr_aucs))
                pr_std = float(np.std(fold_pr_aucs))

                tn, fp, fn, tp = confusion_matrix(
                    y_true, y_pred, labels=[0, 1]
                ).ravel()

                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                # Add model name to fold_ops for later aggregation
                for op in fold_ops:
                    op["model"] = model_name
                all_fold_ops.extend(fold_ops)

                results_by_timescale[t][model_name] = {
                    "auc": auc_mean,
                    "auc_std": auc_std,
                    "pr_auc": pr_mean,
                    "pr_auc_std": pr_std,
                    "n_folds": len(fold_aucs),
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn),
                    "y_true": y_true,
                    "y_prob": y_prob,
                    "y_pred": y_pred,
                    "fold_ops": fold_ops,  # NEW: store fold-level ops
                }

                log.info(
                    f"    {model_name:20s}: "
                    f"AUC={auc_mean:.3f}±{auc_std:.3f}, "
                    f"PR={pr_mean:.3f}±{pr_std:.3f}, "
                    f"Recall={recall:.3f}"
                )

            except Exception as e:
                log.warning(f"    {model_name}: Failed - {e}")

    # Store all fold ops in results for later aggregation
    results_by_timescale["_fold_ops"] = all_fold_ops

    return results_by_timescale

#figure 1
def plot_timescale_horizon(results_dict, output="figure1_timescale_horizon.png",
                           best_model = None):
    plt.figure(figsize=(6.5, 4.8))

    colors = {
        "protein_logo": "#2ca02c",
        "pocket_gkf": "#d62728",
    }

    for key, results in results_dict.items():
        times, roc_vals, pr_vals , roc_stds, pr_stds = [], [], [], [], []

        for t in [1,2, 3, 4, 5, 6, 7, 8, 10, 12,14]:
            if t not in results:
                continue
            if not best_model:
                best_model = select_best_model(results[t], metric="auc")

            #best_model = "Logistic Regression"  # frozen family

            if best_model is None:
                continue

            r = results[t][best_model]
            print(f"Best_model = {best_model}")
            times.append(t)
            roc_vals.append(r["auc"])
            roc_stds.append(r["auc_std"])
            pr_vals.append(r["pr_auc"])
            pr_stds.append(r["pr_auc_std"])
            print(
                f"{key}  t={t}  best={best_model}  "
                f"AUC={r['auc']:.3f}±{r['auc_std']:.3f}  "
                f"PR={r['pr_auc']:.3f}±{r['pr_auc_std']:.3f}"
            )
        label = (
            "Cross-protein"
            if key == "protein_logo"
            else "Cross-pocket"
        )

        plt.plot(times, roc_vals, marker="o", linewidth=1.0,
                 color=colors[key], label=f"{label} – ROC")
        plt.plot(times, pr_vals, marker="s", linestyle="--", linewidth=1.0,
                 color=colors[key], label=f"{label} – PR")

        # mean = np.array(roc_vals)
        # std  = np.array(roc_stds)
        # plt.plot(times, mean, marker="o", linewidth=.5, label=label)
        # mask = np.array(times) <= 7
        # plt.fill_between(
        #     np.array(times)[mask],
        #     (mean - std)[mask],
        #     (mean + std)[mask],
        #     alpha=0.11
        # )

    plt.axvspan(2, 7, color="gray", alpha=0.12)
    plt.text(6.0, 0.85, "Predictive horizon\n(2–7 ns)",
             ha="center", va="top", fontsize=10, style="italic", color="#444")

    plt.xlabel("Early MD Simulation Time (ns)")
    plt.ylabel("Predictive performance")
    plt.title("")
    plt.ylim(0.3, 0.90)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


from sklearn.metrics import roc_curve

def recall_at_fpr(y_true, y_prob, target_fpr=0.2):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid = tpr[fpr <= target_fpr]
    return valid.max() if len(valid) > 0 else 0.0

def select_best_model(results_at_timescale, metric="auc"):
    """
    Select the best model for a single timescale based on a metric.
    Default: highest ROC-AUC.
    Returns model_name or None.
    """
    best_model = None
    best_score = -np.inf

    for model_name, r in results_at_timescale.items():
        if metric in r and r[metric] is not None:
            if r[metric] > best_score:
                best_score = r[metric]
                best_model = model_name

    return best_model

def plot_triage_efficiency(results_dict, output="figure2_triage_efficiency.png",
                           fpr_targets=(0.10, 0.20)):
    plt.figure(figsize=(6.5, 4.5))

    styles = {
        0.20: dict(linestyle="-",  marker="o"),
        0.10: dict(linestyle="--", marker="s"),
    }

    for key, results in results_dict.items():
        label_base = "Cross-protein" if key == "protein_logo" else "Cross-pocket"

        for fpr_t in fpr_targets:
            times, recalls = [], []

            for t in TIMESCALES:
                if t not in results:
                    continue

                # choose best model at this timescale for THIS fpr_t
                best_model = None
                best_recall = -1

                for model_name, r in results[t].items():
                    rec = recall_at_fpr(r["y_true"], r["y_prob"], fpr_t)
                    if rec > best_recall:
                        best_recall = rec
                        best_model = model_name

                if best_model is None:
                    continue

                times.append(t)
                recalls.append(best_recall)

            st = styles.get(fpr_t, dict(linestyle="-", marker="o"))
            plt.plot(times, recalls, linewidth=2.5,
                     label=f"{label_base} (best, FPR ≤ {int(fpr_t*100)}%)",
                     **st)

    plt.axvspan(2, 7, color="gray", alpha=0.15, label="Early MD (2–7 ns)")
    plt.xlabel("Early MD Simulation Time (ns)")
    plt.ylabel("Recall")
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def plot_triage_efficiency_bar(
        results_dict,
        output="figure2_triage_efficiency.png",
        target_fpr=0.2,
):
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7.0, 4.5))

    times = [t for t in TIMESCALES if 2 <= t <= 14]

    # Bar layout
    bar_width = 0.35
    x = np.arange(len(times))

    for i, (key, results) in enumerate(results_dict.items()):
        recalls = []

        for t in times:
            if t not in results:
                recalls.append(np.nan)
                continue

            best_model = select_best_model(results[t], metric="auc")
            if best_model is None:
                recalls.append(np.nan)
                continue

            r = results[t][best_model]
            recall = recall_at_fpr(r["y_true"], r["y_prob"], target_fpr)
            recalls.append(recall)

        label = (
            "Cross-protein"
            if key == "protein_logo"
            else "Cross-pocket"
        )

        plt.bar(
            x + i * bar_width,
            recalls,
            width=bar_width,
            label=label,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.6,
            )

    # Highlight early-screening regime
    early_idx = [i for i, t in enumerate(times) if 2 <= t <= 7]
    if early_idx:
        plt.axvspan(
            min(early_idx) - 0.5,
            max(early_idx) + 0.5,
            color="gray",
            alpha=0.12,
            label="Early MD (2–7 ns)",
            )

    plt.xticks(x + bar_width / 2, times)
    plt.xlabel("MD time window (ns)")
    plt.ylabel(f"Recall at FPR = {target_fpr}")
    plt.title("")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    log.info(f"✓ Saved {output}")

def plot_early_time_separation_compact(
        features_path: Path,
        timescale=2,
        output="early_separation_compact_2ns.png"
):
    df = pd.read_csv(features_path)
    df = df[df["time_ns"] == timescale].copy()
    df = df[df["label_unstable"].notna()]
    df["Stability"] = df["label_unstable"].map({0: "Stable", 1: "Unstable"})

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()

    FEATURE_LABELS = {
        "slope": "RMSD Drift (β)",
        "rmsd_var": "RMSD Variance",
        "mean_disp": "Mean Cα Displacement",
        "var_disp": "Cα Displacement Variance",
    }
    for ax, feat in zip(axes, FEATURE_COLS):

        sns.violinplot(
            data=df,
            x="Stability",
            y=feat,
            palette=["#4C72B0", "#DD8452"],
            cut=0,
            ax=ax
        )


        ax.set_title(FEATURE_LABELS.get(feat, feat))
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    log.info(f"✓ Saved {output}")

from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr, mannwhitneyu


def generate_feature_importance_table_for_model(
        features_path: Path,
        timescale: float,
        model_name: str,
        output_csv: Path,
        group_by: str = 'protein'
):
    """
    Compute feature importance for the selected best model at the selected timescale.
    Uses permutation importance (AUC drop) for comparability across models.
    For LR adds standardized coefficients; for tree models adds feature_importances_.
    """
    log.info(f"\nGenerating feature importance at {timescale}ns using {model_name}")

    # Load data
    df = pd.read_csv(features_path)
    df = df[df['label_unstable'].notna()]
    df_t = df[df['time_ns'] == timescale].copy()

    X = df_t[FEATURE_COLS].values
    y = df_t['label_unstable'].values

    # Groups only used for reporting/consistency; importance is fit on all data
    if group_by == 'pocket':
        groups = (df_t['protein'].astype(str) + '|' + df_t['pocket'].astype(str)).values
    else:
        groups = df_t['protein'].values

    # Standardize (keep consistent with training)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Build the same model from your pool
    models_dict = get_models()
    if model_name not in models_dict:
        raise ValueError(f"Unknown model_name={model_name}. Must be one of {list(models_dict.keys())}")

    model = clone(models_dict[model_name])
    model.fit(X_scaled, y)

    # Permutation importance (AUC drop) — comparable across models
    perm = permutation_importance(
        model,
        X_scaled,
        y,
        scoring='roc_auc',
        n_repeats=50,
        random_state=42
    )

    # Optional grounding: correlation with rmsd_late_20ns if present
    has_rmsd_late = 'rmsd_late_20ns' in df_t.columns

    rows = []
    for i, feat in enumerate(FEATURE_COLS):
        rho = np.nan
        if has_rmsd_late:
            rho, _ = spearmanr(df_t[feat], df_t['rmsd_late_20ns'])

        row = {
            "feature": feat,
            "perm_auc_drop": perm.importances_mean[i],
            "perm_auc_drop_std": perm.importances_std[i],
            "spearman_rmsd_late": rho
        }

        # Model-specific importance
        if hasattr(model, "coef_"):  # Logistic Regression
            row["coef"] = float(model.coef_[0][i])
            row["coef_abs"] = float(abs(model.coef_[0][i]))
        if hasattr(model, "feature_importances_"):  # RF / GB
            row["tree_importance"] = float(model.feature_importances_[i])

        rows.append(row)

    importance_df = pd.DataFrame(rows)

    # Sort by permutation importance (primary, comparable across models)
    importance_df = importance_df.sort_values("perm_auc_drop", ascending=False)

    importance_df.to_csv(output_csv, index=False)

    log.info(f"✓ Saved feature importance table: {output_csv}")
    log.info("Top drivers (by permutation ΔAUC):")
    for _, r in importance_df.head(10).iterrows():
        log.info(f"  {r['feature']:12s} | ΔAUC={r['perm_auc_drop']:.3f}")

    return importance_df


def _threshold_at_target_fpr(y_true, y_prob, target_fpr: float):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return (np.nan, np.nan, np.nan)

    fpr, tpr, thr = roc_curve(y_true, y_prob)

    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return (np.nan, np.nan, np.nan)

    # Use as much FPR budget as possible
    best_fpr_idx = valid[np.argmax(fpr[valid])]

    # If multiple thresholds share this FPR, choose highest TPR
    same_fpr = valid[np.isclose(fpr[valid], fpr[best_fpr_idx], atol=1e-6)]
    best = same_fpr[np.argmax(tpr[same_fpr])]

    return float(thr[best]), float(tpr[best]), float(fpr[best])



def _triage_rate_and_actual_fpr(y_true, y_prob, thr: float):
    """
    Given a threshold, return:
      triage_rate = P(pred_unstable)  (how many are rejected early)
      actual_fpr  = FP / (FP + TN)
    """
    if not np.isfinite(thr):
        return (np.nan, np.nan)

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)

    triage_rate = float(np.mean(y_pred == 1))

    # confusion_matrix: rows=true (0,1), cols=pred (0,1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    actual_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else np.nan
    return (triage_rate, actual_fpr)


def _cost_saved(triage_rate: float, t_ns: float, long_ns: float):
    """
    Simple single-decision policy: run t_ns for everyone, then stop early for predicted-unstable.
    Baseline always runs long_ns.
    """
    if not np.isfinite(triage_rate):
        return np.nan
    return float(triage_rate * (long_ns - t_ns) / long_ns)


# ============================================================================
# TABLE III: Operating Point Aggregation Functions
# ============================================================================

def summarize_operating_points(fold_ops, model_name="Logistic Regression"):
    """
    Aggregate fold-level operating point metrics for Table III.

    Args:
        fold_ops: list of dicts with keys: fold, time_ns, fpr_target, recall,
                  achieved_fpr, triage_rate, cost_saved, model
        model_name: which model to filter for (default: Logistic Regression)

    Returns:
        dict: (time_ns, fpr_target) -> {recall_mean, recall_std, triage_mean, ...}
    """
    summary = {}

    # Filter to selected model
    ops = [r for r in fold_ops if r.get("model") == model_name]

    # Get unique (time_ns, fpr_target) combinations
    combos = set((r["time_ns"], r["fpr_target"]) for r in ops if r["time_ns"] is not None)

    for (t_ns, fpr) in sorted(combos):
        rows = [r for r in ops if r["time_ns"] == t_ns and r["fpr_target"] == fpr]

        if len(rows) == 0:
            continue

        # Use sample std (ddof=1) for proper fold-level reporting
        n = len(rows)
        summary[(t_ns, fpr)] = {
            "n_folds": n,
            "recall_mean": np.mean([r["recall"] for r in rows]),
            "recall_std": np.std([r["recall"] for r in rows], ddof=1) if n > 1 else 0.0,
            "triage_mean": np.mean([r["triage_rate"] for r in rows]),
            "triage_std": np.std([r["triage_rate"] for r in rows], ddof=1) if n > 1 else 0.0,
            "cost_mean": np.mean([r["cost_saved"] for r in rows]),
            "cost_std": np.std([r["cost_saved"] for r in rows], ddof=1) if n > 1 else 0.0,
            "fpr_mean": np.mean([r["achieved_fpr"] for r in rows]),
            "fpr_std": np.std([r["achieved_fpr"] for r in rows], ddof=1) if n > 1 else 0.0,
        }

    return summary

# Convenience function to format a single metric as median[IQR]
def format_median_iqr(values, decimals=1):
    """
    Format an array of values as median[Q1, Q3].

    Args:
        values: array-like of numeric values
        decimals: number of decimal places

    Returns:
        str: formatted string like "60.8 [32.1, 89.4]"
    """
    values = np.array(values)
    median = np.median(values)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    return f"{median:.{decimals}f} [{q1:.{decimals}f}, {q3:.{decimals}f}]"

def export_screening_report_csv(
        results_dict: dict,
        output_csv: str,
        long_ns: float = 20.0,
        fpr_targets=(0.10, 0.20),
):
    """
    results_dict can be:
      - results_by_timescale (dict[t][model] = metrics)
      - OR dict[strategy_name] -> results_by_timescale (when group_by == 'both')
    """
    rows = []

    # Normalize input shape:
    # If values look like dict[timescale], treat as single strategy.
    is_multi_strategy = True
    if len(results_dict) > 0:
        any_key = next(iter(results_dict.keys()))
        # crude check: if top-level keys are timescales (numbers), it's single strategy
        if isinstance(any_key, (int, float, np.integer, np.floating)):
            is_multi_strategy = False

    strategies = results_dict.items() if is_multi_strategy else [("single", results_dict)]

    for strategy_name, by_time in strategies:
        # Try to parse "protein_logo" or "pocket_gkf"
        group_by = None
        cv_mode = None
        if strategy_name != "single" and "_" in strategy_name:
            parts = strategy_name.split("_")
            if len(parts) >= 2:
                group_by, cv_mode = parts[0], parts[1]

        # Filter to only numeric timescale keys (skip '_fold_ops' and other metadata)
        timescale_keys = [k for k in by_time.keys() if isinstance(k, (int, float, np.integer, np.floating))]

        for t_ns in sorted(timescale_keys):
            for model_name, r in by_time[t_ns].items():
                y_true = r.get("y_true", None)
                y_prob = r.get("y_prob", None)

                # Some runs may not store y_prob/y_true if a model failed
                if y_true is None or y_prob is None:
                    continue

                row = {
                    "strategy": strategy_name,
                    "group_by": group_by,
                    "cv_mode": cv_mode,
                    "timescale_ns": float(t_ns),
                    "model": model_name,
                    # keep your standard metrics if present
                    "auc": r.get("auc", np.nan),
                    "pr_auc": r.get("pr_auc", np.nan),
                    "recall": r.get("recall", np.nan),
                    "precision": r.get("precision", np.nan),
                    "f1": r.get("f1", np.nan),
                }

                for fpr_t in fpr_targets:
                    thr, tpr_at, fpr_at = _threshold_at_target_fpr(y_true, y_prob, target_fpr=fpr_t)
                    triage_rate, actual_fpr = _triage_rate_and_actual_fpr(y_true, y_prob, thr)
                    cost_saved = _cost_saved(triage_rate, t_ns=float(t_ns), long_ns=float(long_ns))

                    # columns requested
                    row[f"recall@fpr{fpr_t:.2f}"] = tpr_at            # == TPR at threshold
                    row[f"triage_rate@fpr{fpr_t:.2f}"] = triage_rate
                    row[f"cost_saved@fpr{fpr_t:.2f}"] = cost_saved

                    # useful debugging columns (optional but nice)
                    row[f"thr@fpr{fpr_t:.2f}"] = thr
                    row[f"actual_fpr@thr(fpr{fpr_t:.2f})"] = actual_fpr

                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df

def generate_comparison_figure(results_dict, output_path, features_path):
    """Generate figure comparing protein-level vs pocket-level CV."""

    log.info("\n" + "="*80)
    log.info("GENERATING COMPARISON FIGURE")
    log.info("="*80)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'protein_logo': 'steelblue', 'pocket_gkf': 'darkgreen'}
    labels = {'protein_logo': 'Leave-one-protein-out', 'pocket_gkf': 'Pocket-level GroupKFold'}

    # Panel 1: AUC comparison
    ax = axes[0]
    for key, results in results_dict.items():
        aucs = []
        times = []
        for t in TIMESCALES:
            if t not in results:
                continue

            best_model = select_best_model(results[t], metric="auc")
            if best_model is None:
                continue

            aucs.append(results[t][best_model]['auc'])
            times.append(t)


        if aucs:
            ax.plot(times, aucs, marker='o', linewidth=2.5, markersize=8,
                    label=labels[key], color=colors[key], alpha=0.8)

    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Random')
    ax.set_xlabel('Simulation Time (ns)', fontsize=13)
    ax.set_ylabel('ROC-AUC (Logistic Regression)', fontsize=13)
    ax.set_title('Model Performance: Protein vs Pocket CV', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])

    # Panel 2: Performance table at 2ns
    ax2 = axes[1]
    ax2.axis('off')

    # Panel 2: Performance table at 2ns
    table_data = [['CV Strategy', '2ns AUC', 'Recall', 'Precision', 'Groups']]

    for key, results in results_dict.items():
        if 2 not in results:
            continue

        best_model = select_best_model(results[2], metric="auc")
        if best_model is None:
            continue

        r = results[2][best_model]

        df = pd.read_csv(features_path)
        df_2ns = df[df['time_ns'] == 2.0]
        if key == 'protein_logo':
            n_groups = df_2ns['protein'].nunique()
        else:
            n_groups = (df_2ns['protein'].astype(str) + '|' +
                        df_2ns['pocket'].astype(str)).nunique()

        table_data.append([
            labels[key],
            f"{r['auc']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['precision']:.3f}",
            str(n_groups)
        ])


    table = ax2.table(cellText=table_data, cellLoc='left', loc='center',
                      bbox=[0, 0.3, 1, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4C72B0')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax2.set_title('2ns Performance Comparison', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    log.info(f"✓ Saved comparison figure: {output_path}")
    plt.close()

def load_results_from_pkl(pkl_path: Path):
    log.info(f"✓ Loading results from {pkl_path}")
    with open(pkl_path, "rb") as f:
        results_dict = pickle.load(f)

    required = {"protein_logo", "pocket_gkf"}
    missing = required - set(results_dict.keys())
    if missing:
        raise ValueError(f"Results file missing keys: {missing}")

    return results_dict

def compute_dataset_stats(features_path: Path, timescales=None):
    if timescales is None:
        timescales = [1]
    df = pd.read_csv(features_path)
    df = df[df['label_unstable'].notna()]

    rows = []
    for t in timescales:
        df_t = df[df['time_ns'] == t]
        n_total = len(df_t)
        n_unstable = int(df_t['label_unstable'].sum())
        n_stable = n_total - n_unstable

        rows.append({
            "timescale_ns": t,
            "n_total": n_total,
            "n_stable": n_stable,
            "n_unstable": n_unstable,
            "unstable_frac": n_unstable / n_total if n_total > 0 else np.nan
        })

    return pd.DataFrame(rows)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def _metrics_from_y_pred(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    triage_rate = float(np.mean(np.array(y_pred) == 1))  # fraction flagged for early stop
    actual_fpr = fp / (fp + tn) if (fp + tn) else np.nan
    return recall, triage_rate, actual_fpr

def _cost_saved(triage_rate, t_ns, long_ns=20.0):
    # everyone pays t_ns; triaged ones stop and save (long_ns - t_ns)
    if not np.isfinite(triage_rate):
        return np.nan
    return float(triage_rate * (long_ns - t_ns) / long_ns)

def _metrics_from_fold_ops(fold_ops, fpr_target):
    """
    Aggregate operating-point metrics at a given fpr_target using fold_ops.
    Uses pooled counts across folds (more stable than unweighted averaging).
    """
    if not fold_ops:
        return np.nan, np.nan, np.nan

    ops = [op for op in fold_ops if float(op.get("fpr_target", -1)) == float(fpr_target)]
    if not ops:
        return np.nan, np.nan, np.nan

    tp = sum(int(op.get("tp", 0)) for op in ops)
    fn = sum(int(op.get("fn", 0)) for op in ops)
    fp = sum(int(op.get("fp", 0)) for op in ops)
    tn = sum(int(op.get("tn", 0)) for op in ops)
    n_test = sum(int(op.get("n_test", 0)) for op in ops)

    recall = tp / (tp + fn + 1e-8)
    triage_rate = (tp + fp) / (n_test + 1e-8)
    achieved_fpr = fp / (fp + tn + 1e-8)
    return float(recall), float(triage_rate), float(achieved_fpr)

def plot_triage_and_cost_frozen_model(
        results_by_strategy_fpr,          # dict: target_fpr -> results_dict
        frozen_model_by_strategy,         # dict: strategy_key -> model_name
        TIMESCALES,
        long_ns=20.0,
        out_recall="triage_recall_frozen.png",
        out_cost=f"triage_cost_frozen.png",
):
    labels = {'protein_logo': 'Cross-protein CV', 'pocket_gkf': 'Cross-pocket CV'}
    colors = {'protein_logo': '#4C72B0', 'pocket_gkf': '#55A868'}

    # Style: FPR=0.20 is solid 'o-', FPR=0.10 is dashed 's--'
    styles = {
        (0.20, "protein_logo"): dict(linestyle="-",  marker="o", color=colors['protein_logo'], linewidth=2, markersize=8),
        (0.20, "pocket_gkf"):   dict(linestyle="-",  marker="o", color=colors['pocket_gkf'],   linewidth=2, markersize=8),
        (0.10, "protein_logo"): dict(linestyle="--", marker="s", color=colors['protein_logo'], linewidth=1.5, markersize=6, alpha=0.7),
        (0.10, "pocket_gkf"):   dict(linestyle="--", marker="s", color=colors['pocket_gkf'],   linewidth=1.5, markersize=6, alpha=0.7),
    }


    # -------- Recall plot --------
    plt.figure(figsize=(6.5, 4.5))
    for target_fpr, results_dict in results_by_strategy_fpr.items():
        for strategy_key, by_time in results_dict.items():
            model_name = frozen_model_by_strategy[strategy_key]
            fpr_label = f"FPR≤{int(target_fpr*100)}%"

            xs, ys = [], []
            for t in TIMESCALES:
                if t not in by_time:
                    continue
                r = by_time[t].get(model_name)
                if not r or "fold_ops" not in r:
                    continue

                recall, _, _ = _metrics_from_fold_ops(r["fold_ops"], target_fpr)
                if np.isfinite(recall):
                    xs.append(t)
                    ys.append(recall * 100)  # Convert to percentage

            st = styles.get((float(target_fpr), strategy_key), {})
            plt.plot(xs, ys, label=f"{labels[strategy_key]} ({fpr_label})", **st)

    plt.axvspan(2, 7, alpha=0.15)
    plt.xlabel("Early MD Simulation Time (ns)")
    plt.ylabel("Recall (%)")
    plt.ylim(0, 100)
    plt.xlim(0, 15)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    plt.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.savefig(out_recall, dpi=300, bbox_inches="tight")
    plt.close()

    colors = {'protein_logo': '#4C72B0', 'pocket_gkf': '#55A868'}
    # Style: FPR=0.20 is solid 'o-', FPR=0.10 is dashed 's--'
    styles = {
        (0.20, "protein_logo"): dict(linestyle="-",  marker="o", color=colors['protein_logo'], linewidth=2, markersize=8),
        (0.20, "pocket_gkf"):   dict(linestyle="-",  marker="o", color=colors['pocket_gkf'],   linewidth=2, markersize=8),
        (0.10, "protein_logo"): dict(linestyle="--", marker="s", color=colors['protein_logo'], linewidth=1.5, markersize=6, alpha=0.7),
        (0.10, "pocket_gkf"):   dict(linestyle="--", marker="s", color=colors['pocket_gkf'],   linewidth=1.5, markersize=6, alpha=0.7),
    }
    # -------- Cost-saved plot --------
    plt.figure(figsize=(6.5, 4.5))
    for target_fpr, results_dict in results_by_strategy_fpr.items():
        for strategy_key, by_time in results_dict.items():
            model_name = frozen_model_by_strategy[strategy_key]
            fpr_label = f"FPR≤{int(target_fpr*100)}%"

            xs, ys = [], []
            for t in TIMESCALES:
                if t not in by_time:
                    continue
                r = by_time[t].get(model_name)
                if not r or "fold_ops" not in r:
                    continue

                _, triage_rate, _ = _metrics_from_fold_ops(r["fold_ops"], target_fpr)
                cs = _cost_saved(triage_rate, t_ns=t, long_ns=long_ns)
                if np.isfinite(cs):
                    xs.append(t)
                    ys.append(cs * 100)  # Convert to percentage

            st = styles.get((float(target_fpr), strategy_key), {})
            plt.plot(xs, ys, label=f"{labels[strategy_key]} ({fpr_label})", **st)

    plt.axvspan(2, 7, alpha=0.15)
    plt.xlabel("Early MD Simulation Time (ns)")
    plt.ylabel("Computational Cost Saved (%)")
    plt.ylim(0, 55)
    plt.xlim(0, 15)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    plt.axhline(0, color='gray', linestyle='-', alpha=0.3)
    plt.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.savefig(out_cost, dpi=300, bbox_inches="tight")
    plt.close()

    # -------- Print summary table --------
    print("\n" + "="*90)
    print("TRIAGE PERFORMANCE SUMMARY")
    print("="*90)
    print(f"{'Strategy':<18} {'FPR':<6} {'t(ns)':<6} {'Recall%':<10} {'CostSaved%':<12} {'AUC±std':<16} {'PR-AUC±std':<16}")
    print("-"*90)

    for target_fpr in sorted(results_by_strategy_fpr.keys()):
        results_dict = results_by_strategy_fpr[target_fpr]
        for strategy_key in ['protein_logo', 'pocket_gkf']:
            if strategy_key not in results_dict:
                continue
            by_time = results_dict[strategy_key]
            model_name = frozen_model_by_strategy[strategy_key]

            for t in TIMESCALES:
                if t not in by_time:
                    continue
                r = by_time[t].get(model_name)
                if not r or "y_true" not in r or "y_pred" not in r:
                    continue

                recall, triage_rate, _ = _metrics_from_fold_ops(r["fold_ops"], target_fpr)
                cs = _cost_saved(triage_rate, t_ns=t, long_ns=long_ns)
                auc = r.get("auc", np.nan)
                auc_std = r.get("auc_std", np.nan)
                pr_auc = r.get("pr_auc", np.nan)
                pr_auc_std = r.get("pr_auc_std", np.nan)

                print(f"{labels[strategy_key]:<18} {target_fpr:<6.2f} {t:<6} "
                      f"{recall*100:>6.1f}%    {cs*100:>6.1f}%      "
                      f"{auc:.3f}±{auc_std:.3f}      {pr_auc:.3f}±{pr_auc_std:.3f}")
        print("-"*90)

    print("="*90 + "\n")

def main():
    date_tag = "20251225"
    version = "1.1"
    parser = argparse.ArgumentParser(
        description="Generate comprehensive ML analysis figure with different CV strategies"
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(f"outputs/fingerprint_summary_with_components_even_ac_4d_drift_{date_tag}.csv"),
        help="Path to fingerprint_summary_with_components.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"./outputs/comprehensive_ml_both_cv_{date_tag}.png"),
        help="Output figure path"
    )
    parser.add_argument(
        "--timescales",
        type=float,
        nargs="+",
        default=TIMESCALES,
        help="Timescales to evaluate (default: )"
    )
    parser.add_argument(
        "--group_by",
        type=str,
        default='both',
        choices=['protein', 'pocket', 'ligand_pocket', 'both'],
        help="Grouping strategy: 'protein' (protein-level), 'pocket' (pocket-level), 'ligand_pocket' (protein|ligand|pocket), or 'both'"
    )
    parser.add_argument(
        "--cv_mode",
        type=str,
        default='logo',
        choices=['logo', 'gkf'],
        help="CV mode: 'logo' (LeaveOneGroupOut) or 'gkf' (GroupKFold)"
    )
    parser.add_argument(
        "--load_results",
        type=Path,
        default=Path(f'./outputs/results_both_cv_{date_tag}_{version}.pkl') ,
        help="Path to existing results .pkl file (skip training if provided)"
    )
    parser.add_argument(
        "--plot_only",
        default= True,
        help="Only generate plots from existing results (no training)"
    )

    parser.add_argument(
        "--print_fold_summary",

        default=False,
        help="Print per-fold test-set sizes and class counts at a selected timescale (useful for LOPO small-n interpretation)."
    )
    parser.add_argument(
        "--fold_summary_timescale",
        type=float,
        default=2.0,
        help="Timescale (ns) at which to print per-fold CV test-set sizes when --print_fold_summary is set (default: 2)."
    )

    args = parser.parse_args()

    log.info("="*80)
    log.info("COMPREHENSIVE ML FIGURE GENERATION WITH MULTIPLE CV STRATEGIES")
    log.info("="*80)

    results_dict = {}
    if args.plot_only:
        if args.load_results is None:
            raise ValueError("--plot_only requires --load_results")

        results_dict = load_results_from_pkl(args.load_results)
        generate_comparison_figure(results_dict, args.output, args.features)

    elif args.group_by == 'both':
        # Run both strategies for comparison
        log.info("\n" + "="*80)
        log.info("EVALUATION 1: Leave-one-protein-out (protein generalization)")
        log.info("="*80)

        results_protein = evaluate_all_models(args.features, args.timescales,
                                              group_by='protein', cv_mode='logo',
                                              print_fold_summary=args.print_fold_summary,
                                              fold_summary_timescale=args.fold_summary_timescale)
        results_dict['protein_logo'] = results_protein

        log.info("\n" + "="*80)
        log.info("EVALUATION 2: Pocket-level GroupKFold (pocket generalization)")
        log.info("="*80)

        results_pocket = evaluate_all_models(args.features, args.timescales,
                                             group_by='ligand_pocket', cv_mode='gkf',
                                             print_fold_summary=args.print_fold_summary,
                                             fold_summary_timescale=args.fold_summary_timescale)
        results_dict['pocket_gkf'] = results_pocket
        # Generate comparison figure
        generate_comparison_figure(results_dict, args.output, args.features)
        results_dict = {
            "protein_logo": results_protein,
            "pocket_gkf": results_pocket,
        }

        with open(f"./outputs/results_both_cv_{date_tag}_{version}.pkl", "wb") as f:
            pickle.dump(results_dict, f)

        log.info("✓ Saved combined results: results_both_cv.pkl")
        # After results_dict is populated:
        report_path = f"./outputs/screening_report_both_cv_{date_tag}_{version}.csv" if args.group_by == "both" \
            else f"./outputs/screening_report_{args.group_by}_{args.cv_mode}_{date_tag}_{version}.csv"

        export_screening_report_csv(
            results_dict=results_dict,
            output_csv=report_path,
            long_ns=20.0,
            fpr_targets=(0.10, 0.20),
        )
        log.info(f"✓ Saved screening report CSV: {report_path}")

    compute_dataset_stats(features_path=args.features)

    dataset_stats = compute_dataset_stats(features_path=args.features)
    dataset_stats.to_csv(
        f"./outputs/dataset_size_by_timescale_{date_tag}_{version}.csv",
        index=False
    )
    plot_timescale_horizon(
        results_dict,
        output=f"./outputs/figure1_timescale_horizon_{date_tag}_{version}.png",
        best_model = "Logistic Regression"
    )


    plot_triage_efficiency(
        results_dict,
        output=f"./outputs/figure2_triage_efficiency_{date_tag}_{version}.png"
    )

    plot_early_time_separation_compact(
        features_path=args.features,
        timescale=2,
        output= f"./outputs/early_time_separation_2ns_{date_tag}_{version}.png"
    )

    # strategy_key examples in your CSV: "protein_logo" and "pocket_gkf"
    # Check for cached FPR results
    fpr_results_path = Path(f"./outputs/results_fpr_strategies_{date_tag}_{version}.pkl")

    if fpr_results_path.exists():
        log.info(f"Loading cached FPR results from {fpr_results_path}")
        with open(fpr_results_path, "rb") as f:
            results_by_strategy_fpr = pickle.load(f)
    else:
        log.info("Computing FPR results (this may take a while)...")

        # FIX C: Run evaluate_all_models ONCE per strategy with both FPR targets
        # Each call already computes fold_ops for all fpr_targets=[0.10, 0.20]
        # We use GOAL_FPR=0.10 for threshold selection during CV, but fold_ops
        # contains metrics for BOTH FPR targets

        results_protein = evaluate_all_models(
            args.features, args.timescales,
            group_by="protein", cv_mode="logo",
            GOAL_FPR=0.10,  # Default threshold for y_pred
            fpr_targets=(0.10, 0.20),  # Both targets computed in fold_ops
            print_fold_summary=args.print_fold_summary,
            fold_summary_timescale=args.fold_summary_timescale,
        )

        results_pocket = evaluate_all_models(
            args.features, args.timescales,
            group_by="ligand_pocket", cv_mode="gkf",
            GOAL_FPR=0.10,
            fpr_targets=(0.10, 0.20),
            print_fold_summary=args.print_fold_summary,
            fold_summary_timescale=args.fold_summary_timescale,
        )

        # Structure results by FPR target for backward compatibility with plotting functions
        # Note: The y_pred in results is based on GOAL_FPR, but fold_ops has both
        results_by_strategy_fpr = {
            0.10: {
                "protein_logo": results_protein,
                "pocket_gkf": results_pocket,
            },
            0.20: {
                "protein_logo": results_protein,
                "pocket_gkf": results_pocket,
            },
        }

        # Save FPR results for future use
        with open(fpr_results_path, "wb") as f:
            pickle.dump(results_by_strategy_fpr, f)
        log.info(f"✓ Saved FPR results to {fpr_results_path}")

    # Freeze one model per strategy (example choices)
    frozen_model_by_strategy = {
        "protein_logo": "Logistic Regression",
        "pocket_gkf": "Logistic Regression",
    }

    plot_triage_and_cost_frozen_model(
        results_by_strategy_fpr,
        frozen_model_by_strategy,
        TIMESCALES=[1,2,3,4,5,6,7,8,10,12,14],
        long_ns=20.0,
        out_recall=f"./outputs/triage_out_recall_{date_tag}_{version}.png",
        out_cost=f"./outputs/triage_cost_frozen_{date_tag}_{version}.png"

    )

    # ============================================================================
    # Generate Table III: Operating Point Recommendations
    # ============================================================================
    log.info("\n" + "="*80)
    log.info("GENERATING TABLE III: OPERATING POINT RECOMMENDATIONS")
    log.info("="*80)

    # Print operating point summary first
    #print_operating_point_summary(results_by_strategy_fpr, frozen_model_by_strategy)

    # Generate the full Table III with CSV and LaTeX outputs
    # Fix D: Only the 3 recommended operating points for the paper
    table_iii_df = generate_table_iii(
        results_by_strategy_fpr,
        frozen_model_by_strategy,
        operating_points=[(2, 0.20), (3, 0.20), (3, 0.10),  (7, 0.20), (7, 0.10)],  # Paper's recommended modes
        output_csv=f"./outputs/table_iii_operating_points_{date_tag}_{version}.csv",
        output_latex=f"./outputs/table_iii_operating_points_{date_tag}_{version}.tex",
        debug=True
    )

    log.info("\n" + "="*80)
    log.info("COMPLETE!")
    log.info("="*80)



if __name__ == "__main__":
    main()

# Example usage:
# python generate_comprehensive_figure.py \
#     --features fingerprint_summary_with_components.csv \
#     --output comprehensive_ml_results.png