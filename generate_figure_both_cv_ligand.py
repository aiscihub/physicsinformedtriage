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
        )
    }


def train_model_cv(X, y, groups, model, cv_mode='logo', target_fpr=0.10):
    """
    Train model with group-aware cross-validation.
    
    Args:
        cv_mode: 'logo' (LeaveOneGroupOut) or 'gkf' (GroupKFold with 5 splits)
    """
    from sklearn.model_selection import GroupKFold
    
    if cv_mode == 'gkf':
        cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    else:
        cv = LeaveOneGroupOut()
    
    scaler = StandardScaler()

    y_true_all = []
    y_prob_all = []
    y_pred_all = []

    for train_idx, test_idx in cv.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)

        #a probability threshold chosen for recall
        y_prob_train = model.predict_proba(X_train_scaled)[:, 1]
        thr, _, _ = _threshold_at_target_fpr(y_train, y_prob_train, target_fpr=target_fpr)
        # Probabilities on TEST fold
        y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
        y_pred_test = (y_prob_test >= thr).astype(int)


        y_true_all.extend(y_test)
        y_prob_all.extend(y_prob_test)
        y_pred_all.extend(y_pred_test)

    return np.array(y_true_all), np.array(y_prob_all), np.array(y_pred_all)


def evaluate_all_models(features_path: Path, timescales: list,
                        group_by='protein',
                        cv_mode='logo', GOAL_FPR = 0.10):
    """
    Train and evaluate all models at all timescales.
    
    Args:
        group_by: 'protein' or 'pocket' - how to define groups
        cv_mode: 'logo' (LeaveOneGroupOut) or 'gkf' (GroupKFold)
    """

    log.info("="*80)
    log.info("TRAINING MODELS")
    log.info("="*80)
    log.info(f"Grouping strategy: {group_by}")
    log.info(f"CV mode: {cv_mode}")

    results_by_timescale = {}
    models_dict = get_models()

    for t in timescales:
        log.info(f"\nTimescale: {t}ns")
        X, y, groups = load_data(features_path, t, group_by=group_by)



        log.info(f"  Samples: {len(X)}, Unstable: {int(y.sum())} ({100*y.mean():.1f}%)")
        log.info(f"  Unique groups: {len(np.unique(groups))}")

        results_by_timescale[t] = {}

        for model_name, model in models_dict.items():
            try:
                y_true, y_prob, y_pred = train_model_cv(X, y, groups, model, cv_mode=cv_mode, target_fpr=GOAL_FPR)

                if len(np.unique(y_true)) > 1:
                    auc = roc_auc_score(y_true, y_prob)
                    pr_auc = average_precision_score(y_true, y_prob)

                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                    results_by_timescale[t][model_name] = {
                        'auc': auc,
                        'pr_auc': pr_auc,
                        'recall': recall,
                        'precision': precision,
                        'f1': f1,
                        'tp': tp,
                        'fp': fp,
                        'tn': tn,
                        'fn': fn,
                        'y_true': y_true,
                        'y_prob': y_prob,
                        'y_pred': y_pred
                    }

                    log.info(f"    {model_name:20s}: AUC={auc:.3f}, Recall={recall:.3f}")
            except Exception as e:
                log.warning(f"    {model_name}: Failed - {e}")

     # ------------------------------------------------
    # Feature importance (best timescale + best model in pool)
    # ------------------------------------------------
    best_time, best_model, best_score = select_best_time_and_model(results_by_timescale, metric="auc")

    if best_time is not None and best_model is not None:
        generate_feature_importance_table_for_model(
            features_path=features_path,
            timescale=best_time,
            model_name=best_model,
            output_csv=Path(f"./dataset//feature_importance_{group_by}_{cv_mode}_{best_model.replace(' ','_')}.csv"),
            group_by=group_by
        )
    return results_by_timescale
#figure 1
def plot_timescale_horizon(results_dict, output="figure1_timescale_horizon.png"):
    plt.figure(figsize=(6.5, 4.8))

    colors = {
        "protein_logo": "#2ca02c",
        "pocket_gkf": "#d62728",
    }

    for key, results in results_dict.items():
        times, roc_vals, pr_vals = [], [], []

        for t in [1,2, 3, 4, 5, 6, 7, 8, 10, 12,14]:
            if t not in results:
                continue

            best_model = select_best_model(results[t], metric="auc")
            if best_model is None:
                continue

            r = results[t][best_model]
            times.append(t)
            roc_vals.append(r["auc"])
            pr_vals.append(r["pr_auc"])

        label = (
            "Cross-protein"
            if key == "protein_logo"
            else "Cross-pocket"
        )

        plt.plot(times, roc_vals, marker="o", linewidth=2.5,
                 color=colors[key], label=f"{label} – ROC")
        plt.plot(times, pr_vals, marker="s", linestyle="--", linewidth=1.5,
                 color=colors[key], label=f"{label} – PR")

    # plt.axvspan(4.5, 7.5, color="gray", alpha=0.12)
    # plt.text(6.0, 0.83, "Predictive horizon\n(5–7 ns)",
    #          ha="center", va="top", fontsize=10, style="italic", color="#444")

    plt.xlabel("Early MD Simulation Time (ns)")
    plt.ylabel("Predictive performance")
    plt.title("")
    plt.ylim(0.45, 0.90)
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
from scipy.stats import spearmanr


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
    """
    Pick a probability threshold that achieves FPR <= target_fpr while maximizing TPR.
    Returns (thr, tpr_at_thr, fpr_at_thr). If undefined, returns (np.nan, np.nan, np.nan).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    # Need both classes to define ROC
    if len(np.unique(y_true)) < 2:
        return (np.nan, np.nan, np.nan)

    fpr, tpr, thr = roc_curve(y_true, y_prob)  # thr aligned with fpr/tpr

    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return (np.nan, np.nan, np.nan)

    # among valid points, choose the one with max TPR (if ties, highest FPR within constraint)
    best = valid[np.argmax(tpr[valid])]
    return (float(thr[best]), float(tpr[best]), float(fpr[best]))


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

        for t_ns in sorted(by_time.keys()):
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


def main():
    date_tag = "20251225"
    parser = argparse.ArgumentParser(
        description="Generate comprehensive ML analysis figure with different CV strategies"
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(f"dataset/fingerprint_summary_with_components_even_ac_4d_drift_{date_tag}.csv"),
        help="Path to fingerprint_summary_with_components.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./dataset/comprehensive_ml_both_cv.png"),
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
        default=Path(f'./dataset/results_both_cv_{date_tag}.pkl') ,
        help="Path to existing results .pkl file (skip training if provided)"
    )
    parser.add_argument(
        "--plot_only",
        default= False,
        help="Only generate plots from existing results (no training)"
    )

    args = parser.parse_args()
    
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
                                              group_by='protein', cv_mode='logo')
        results_dict['protein_logo'] = results_protein
        
        log.info("\n" + "="*80)
        log.info("EVALUATION 2: Pocket-level GroupKFold (pocket generalization)")
        log.info("="*80)

        results_pocket = evaluate_all_models(args.features, args.timescales,
                                             group_by='ligand_pocket', cv_mode='gkf')
        results_dict['pocket_gkf'] = results_pocket
        # Generate comparison figure
        generate_comparison_figure(results_dict, args.output, args.features)
        results_dict = {
            "protein_logo": results_protein,
            "pocket_gkf": results_pocket,
        }

        with open(f"./dataset/results_both_cv_{date_tag}.pkl", "wb") as f:
            pickle.dump(results_dict, f)

        log.info(f"Saved ./dataset/results_both_cv_{date_tag}.pkl")
        # After results_dict is populated:
        report_path = f"./dataset/screening_report_both_cv_{date_tag}.csv" if args.group_by == "both" \
            else f"./dataset/screening_report_{args.group_by}_{args.cv_mode}_{date_tag}.csv"

        export_screening_report_csv(
            results_dict=results_dict,
            output_csv=report_path,
            long_ns=20.0,
            fpr_targets=(0.10, 0.20),
        )
        log.info(f"✓ Saved screening report CSV: {report_path}")


    log.info("COMPLETE!")

if __name__ == "__main__":
    main()