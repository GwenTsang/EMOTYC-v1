# -*- coding: utf-8 -*-
"""
Visualization — All Plotting Functions
════════════════════════════════════════

Consolidated plotting module. Each function takes data + output path
and produces publication-quality figures. Analytical functions (in other
modules) return data; this module consumes it.

Design: all functions follow the pattern:
    def plot_X(data, out_dir, **kwargs) -> Path
"""

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import config

# ── Style defaults ────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
})


def _ensure_dir(out_dir):
    """Ensure plot directory exists and return it."""
    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


# ═══════════════════════════════════════════════════════════════════════════
#  ERROR DISTRIBUTION PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_error_distributions(df, out_dir, metric="hamming_12"):
    """
    4-panel figure: histogram, boxplot, n_errors distribution,
    Hamming vs Jaccard scatter.
    """
    plot_dir = _ensure_dir(out_dir)
    n_col = f"n_errors_{metric.split('_')[-1]}" if "_" in metric else "n_errors_12"
    j_col = metric.replace("hamming", "jaccard_error")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram
    ax = axes[0, 0]
    ax.hist(df[metric].dropna(), bins=12, alpha=0.8,
            color="steelblue", density=True)
    ax.set_xlabel(f"{metric}")
    ax.set_ylabel("Densité")
    ax.set_title("Distribution de l'erreur")

    # 2. Boxplot
    ax = axes[0, 1]
    sns.boxplot(y=df[metric], ax=ax, color="steelblue")
    ax.set_title(f"Distribution de {metric}")
    ax.set_xlabel("")
    ax.set_ylabel(metric)

    # 3. N_errors distribution
    ax = axes[1, 0]
    if n_col in df.columns:
        error_counts = df[n_col].value_counts().sort_index()
        ax.bar(error_counts.index, error_counts.values,
               color="steelblue", alpha=0.8)
        ax.set_xlabel(f"Nombre d'erreurs")
        ax.set_ylabel("Fréquence")
        ax.set_title("Distribution du nombre d'erreurs par exemple")

    # 4. Hamming vs Jaccard scatter
    ax = axes[1, 1]
    if j_col in df.columns:
        ax.scatter(df[metric], df[j_col],
                   color="steelblue", alpha=0.5, s=15)
        ax.set_xlabel("Hamming Error")
        ax.set_ylabel("Jaccard Error")
        ax.set_title("Hamming vs Jaccard Error")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)

    plt.tight_layout()
    path = plot_dir / "error_distributions.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  PER-LABEL ERROR DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════

def plot_per_label_errors(label_errors_df, df, eval_labels, out_dir):
    """
    Stacked bar (OK/FP/FN per label).
    """
    plot_dir = _ensure_dir(out_dir)

    if label_errors_df is None or len(label_errors_df) == 0:
        return None

    # ── Stacked bar ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = label_errors_df["label"].tolist()
    fp_rates = label_errors_df["FP_rate"].tolist()
    fn_rates = label_errors_df["FN_rate"].tolist()
    ok_rates = label_errors_df["accuracy"].tolist()
    x = np.arange(len(labels))
    width = 0.6

    ax.bar(x, ok_rates, width, label="OK (correct)", color="#4CAF50", alpha=0.8)
    ax.bar(x, fp_rates, width, bottom=ok_rates,
           label="FP (faux positif)", color="#FF9800", alpha=0.8)
    ax.bar(x, fn_rates, width,
           bottom=[o + f for o, f in zip(ok_rates, fp_rates)],
           label="FN (faux négatif)", color="#F44336", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title("Décomposition FP / FN / OK par label émotionnel")
    ax.legend(loc="upper right")

    for i, row in label_errors_df.iterrows():
        ax.annotate(f"p={row['prevalence']:.2f}", (i, 1.01),
                    ha="center", fontsize=7, color="gray")

    plt.tight_layout()
    path = plot_dir / "per_label_error_decomposition.png"
    fig.savefig(path)
    plt.close(fig)

    print(f"  ✓ per_label_error_decomposition.png")
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  UNIVARIATE PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_univariate(results, df, out_dir, metric="hamming_12"):
    """Box plots for each significant feature."""
    plot_dir = _ensure_dir(out_dir)

    for r in results:
        feat = r["feature"]
        if feat not in df.columns:
            continue

        col = df[feat].copy()
        if feat in config.BINARY_FEATURES:
            col = col.astype(str)
        else:
            col = col.astype(str).replace("nan", "MISSING")

        groups = {}
        for level in col.unique():
            vals = df.loc[col == level, metric].dropna()
            if len(vals) >= 3:
                groups[level] = vals

        if len(groups) < 2:
            continue

        fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.2), 5))
        plot_data = pd.DataFrame({"feature": col, "error": df[metric]})
        plot_data = plot_data[plot_data["feature"].isin(groups.keys())]
        order = sorted(groups.keys(), key=lambda l: groups[l].mean(),
                       reverse=True)
        sns.boxplot(data=plot_data, x="feature", y="error", order=order,
                    ax=ax, palette="RdYlGn_r", showfliers=True)

        for i, level in enumerate(order):
            m = groups[level].mean()
            ax.plot(i, m, "D", color="black", markersize=6, zorder=5)

        p = r["p_value"]
        eta = r["eta_squared"]
        ax.set_title(f"Hamming Error by {feat}\n"
                     f"({r['test']}: p={p:.2e}, η²={eta:.3f})")
        ax.set_xlabel(feat)
        ax.set_ylabel(metric)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        fname = f"univariate_{feat.replace(' ', '_').replace('/', '_')}.png"
        fig.savefig(plot_dir / fname)
        plt.close(fig)

    print(f"  ✓ {len(results)} univariate plots")


# ═══════════════════════════════════════════════════════════════════════════
#  BIVARIATE HEATMAPS
# ═══════════════════════════════════════════════════════════════════════════

def plot_bivariate_heatmaps(interaction_scores, df, out_dir,
                            metric="hamming_12", top_n=10):
    """Heatmaps for top feature interaction pairs."""
    plot_dir = _ensure_dir(out_dir)

    for rank, info in enumerate(interaction_scores[:top_n]):
        f1, f2 = info["f1"], info["f2"]
        col1 = (
            df[f1].astype(str).replace("nan", "MISSING")
            if f1 not in config.BINARY_FEATURES else df[f1].astype(str)
        )
        col2 = (
            df[f2].astype(str).replace("nan", "MISSING")
            if f2 not in config.BINARY_FEATURES else df[f2].astype(str)
        )

        combined = pd.DataFrame({
            "f1": col1, "f2": col2, "error": df[metric]
        }).dropna(subset=["error"])
        pivot_mean = combined.pivot_table(
            values="error", index="f1", columns="f2", aggfunc="mean"
        )
        pivot_count = combined.pivot_table(
            values="error", index="f1", columns="f2", aggfunc="count"
        )

        fig, ax = plt.subplots(
            figsize=(max(7, pivot_mean.shape[1] * 1.5),
                     max(5, pivot_mean.shape[0] * 1.0))
        )

        annot = pivot_mean.copy().astype(str)
        for r in annot.index:
            for c in annot.columns:
                m = pivot_mean.loc[r, c]
                n = pivot_count.loc[r, c] if not pd.isna(
                    pivot_count.loc[r, c]) else 0
                if pd.isna(m):
                    annot.loc[r, c] = ""
                else:
                    annot.loc[r, c] = f"{m:.3f}\n(n={int(n)})"

        sns.heatmap(pivot_mean, annot=annot, fmt="", cmap="RdYlGn_r",
                    vmin=0, ax=ax, linewidths=0.5)
        ax.set_title(f"Hamming Error: {f1} × {f2}\n"
                     f"(range={info['error_range']:.3f})")
        ax.set_xlabel(f2)
        ax.set_ylabel(f1)
        plt.tight_layout()

        fname = (f"bivariate_{rank+1:02d}_{f1}_{f2}"
                 .replace(" ", "_").replace("/", "_"))
        fig.savefig(plot_dir / f"{fname}.png")
        plt.close(fig)

    print(f"  ✓ {min(top_n, len(interaction_scores))} bivariate heatmaps")


# ═══════════════════════════════════════════════════════════════════════════
#  RF + SHAP PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_rf_importance(rf_model, feature_names, out_dir, top_k=25):
    """Random Forest MDI feature importance bar chart."""
    plot_dir = _ensure_dir(out_dir)

    if rf_model is None:
        return None

    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_k = min(top_k, len(feature_names))
    top_idx = sorted_idx[:top_k]

    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.35)))
    ax.barh(range(top_k), importances[top_idx][::-1],
            color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, top_k)))
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feature_names[i] for i in top_idx][::-1])
    ax.set_xlabel("MDI Importance")
    ax.set_title("Random Forest — Feature Importance (MDI)")
    plt.tight_layout()

    path = plot_dir / "rf_feature_importance.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path


def plot_shap_summary(shap_values, X_df, out_dir, top_k=25):
    """SHAP summary and bar plots."""
    plot_dir = _ensure_dir(out_dir)

    if shap_values is None:
        return None

    try:
        import shap

        # Summary plot (beeswarm)
        fig, ax = plt.subplots(figsize=(12, max(8, top_k * 0.35)))
        shap.summary_plot(shap_values, X_df, max_display=top_k, show=False)
        plt.tight_layout()
        plt.savefig(plot_dir / "shap_summary.png", bbox_inches="tight")
        plt.close("all")

        # Bar plot (mean |SHAP|)
        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.35)))
        shap.summary_plot(shap_values, X_df, plot_type="bar",
                          max_display=top_k, show=False)
        plt.tight_layout()
        plt.savefig(plot_dir / "shap_bar.png", bbox_inches="tight")
        plt.close("all")

        print(f"  ✓ shap_summary.png + shap_bar.png")
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  CONDITIONAL ANALYSIS PLOTS  (Objective 1)
# ═══════════════════════════════════════════════════════════════════════════

def plot_conditional_heatmaps(cond_results, out_dir):
    """
    Two heatmaps: Δ_F1(emotion | mode) and Δ_F1(mode | emotion).
    Green = synergy (positive Δ), Red = degradation (negative Δ).
    """
    plot_dir = _ensure_dir(out_dir)

    if not cond_results:
        return

    for key, title, fname in [
        ("delta_f1_emotion_given_mode",
         "Δ F1 (Emotion performance | Mode present in gold)",
         "conditional_emotion_given_mode.png"),
        ("delta_f1_mode_given_emotion",
         "Δ F1 (Mode performance | Emotion present in gold)",
         "conditional_mode_given_emotion.png"),
    ]:
        mat = cond_results.get(key)
        if mat is None or mat.empty:
            continue

        mat_float = mat.astype(float)
        # Drop rows/cols that are all NaN
        mat_float = mat_float.dropna(how="all", axis=0).dropna(how="all", axis=1)

        if mat_float.empty:
            continue

        vmax = max(abs(mat_float.min().min()), abs(mat_float.max().max()), 0.01)

        fig, ax = plt.subplots(
            figsize=(max(8, mat_float.shape[1] * 1.2),
                     max(4, mat_float.shape[0] * 0.8))
        )

        # Annotation with values
        annot = mat_float.copy()
        annot = annot.map(
            lambda v: f"{v:+.3f}" if not np.isnan(v) else ""
        )

        sns.heatmap(mat_float, annot=annot, fmt="", cmap="RdYlGn",
                    center=0, vmin=-vmax, vmax=vmax,
                    ax=ax, linewidths=0.5, cbar_kws={"label": "Δ F1"})
        ax.set_title(title, fontsize=11, pad=12)
        plt.tight_layout()
        fig.savefig(plot_dir / fname)
        plt.close(fig)

    print(f"  ✓ conditional heatmaps")


# ═══════════════════════════════════════════════════════════════════════════
#  INTERACTION MATRIX PLOT  (Objective 2)
# ═══════════════════════════════════════════════════════════════════════════

def plot_interaction_matrix(interaction_results, out_dir):
    """
    Heatmap of interaction effects (emotion × mode).
    Red = conflict (worse than expected), Blue = synergy.
    """
    plot_dir = _ensure_dir(out_dir)

    if not interaction_results:
        return

    mat = interaction_results.get("interaction_effects")
    if mat is None or mat.empty:
        return

    mat_float = mat.astype(float).dropna(how="all", axis=0).dropna(how="all", axis=1)
    if mat_float.empty:
        return

    vmax = max(abs(mat_float.min().min()), abs(mat_float.max().max()), 0.01)

    fig, ax = plt.subplots(
        figsize=(max(7, mat_float.shape[1] * 1.8),
                 max(5, mat_float.shape[0] * 0.6))
    )

    annot = mat_float.map(
        lambda v: f"{v:+.3f}" if not np.isnan(v) else ""
    )

    sns.heatmap(mat_float, annot=annot, fmt="", cmap="RdBu_r",
                center=0, vmin=-vmax, vmax=vmax,
                ax=ax, linewidths=0.5,
                cbar_kws={"label": "Interaction Effect (obs − exp)"})
    ax.set_title("Interaction Effects: Emotion × Mode\n"
                 "(Red=conflict, Blue=synergy)", fontsize=11)
    ax.set_xlabel("Mode d'expression")
    ax.set_ylabel("Émotion")
    plt.tight_layout()
    fig.savefig(plot_dir / "interaction_emotion_mode.png")
    plt.close(fig)

    # ── Error co-occurrence heatmap ───────────────────────────────────
    cooc = interaction_results.get("error_cooccurrence")
    if cooc is not None and not cooc.empty:
        fig, ax = plt.subplots(figsize=(max(8, len(cooc) * 0.6),
                                        max(6, len(cooc) * 0.5)))
        mask = np.triu(np.ones_like(cooc, dtype=bool), k=1)
        sns.heatmap(cooc.astype(float), mask=~mask, cmap="YlOrRd",
                    ax=ax, linewidths=0.3, annot=True, fmt=".3f",
                    annot_kws={"fontsize": 7})
        ax.set_title("Error Co-occurrence Rate (upper triangle)")
        plt.tight_layout()
        fig.savefig(plot_dir / "error_cooccurrence.png")
        plt.close(fig)

    print(f"  ✓ interaction + error co-occurrence plots")


# ═══════════════════════════════════════════════════════════════════════════
#  LOGIT DISTRIBUTION PLOTS  (Objective 3)
# ═══════════════════════════════════════════════════════════════════════════

def plot_logit_distributions(logit_df, df, out_dir):
    """
    For each label, overlaid histograms of predicted probabilities
    conditioned on gold=0 vs gold=1.
    """
    plot_dir = _ensure_dir(out_dir)

    if logit_df is None or logit_df.empty:
        return

    labels = logit_df["label"].tolist()
    n_labels = len(labels)
    ncols = 4
    nrows = (n_labels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 3))
    axes = np.array(axes).flatten() if n_labels > 1 else [axes]

    for idx, label in enumerate(labels):
        ax = axes[idx]
        proba_col = f"proba_{label}"
        if proba_col not in df.columns or label not in df.columns:
            ax.set_visible(False)
            continue

        probas = df[proba_col].values
        golds = df[label].values.astype(int)

        ax.hist(probas[golds == 0], bins=30, alpha=0.6, color="steelblue",
                label="gold=0", density=True)
        if (golds == 1).sum() > 0:
            ax.hist(probas[golds == 1], bins=30, alpha=0.6, color="coral",
                    label="gold=1", density=True)

        row = logit_df[logit_df["label"] == label].iloc[0]
        sep = row.get("logit_separation", np.nan)
        sep_str = f"sep={sep:+.1f}" if not np.isnan(sep) else "N/A"
        ax.set_title(f"{label}\n({sep_str})", fontsize=9)
        ax.set_xlabel("P(y=1)")
        ax.legend(fontsize=7)

    # Hide unused axes
    for idx in range(n_labels, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Probability Distributions: gold=0 vs gold=1", fontsize=13)
    plt.tight_layout()
    fig.savefig(plot_dir / "logit_distributions.png")
    plt.close(fig)
    print(f"  ✓ logit_distributions.png")


def plot_calibration_diagrams(calibration_data, out_dir):
    """Reliability diagrams for emotions and modes."""
    plot_dir = _ensure_dir(out_dir)

    if not calibration_data:
        return

    labels = list(calibration_data.keys())
    n_labels = len(labels)
    ncols = 4
    nrows = (n_labels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.5))
    axes = np.array(axes).flatten() if n_labels > 1 else [axes]

    for idx, label in enumerate(labels):
        ax = axes[idx]
        data = calibration_data[label]
        mids = data["bin_midpoints"]
        obs = data["observed_freq"]
        counts = data["bin_count"]

        # Filter out empty bins
        valid = [(m, o, c) for m, o, c in zip(mids, obs, counts)
                 if c > 0 and not np.isnan(o)]
        if not valid:
            ax.set_visible(False)
            continue

        mid_v, obs_v, cnt_v = zip(*valid)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=0.8)
        ax.bar(mid_v, obs_v, width=0.08, alpha=0.6, color="steelblue")
        ax.plot(mid_v, obs_v, "ro-", markersize=3, linewidth=1)

        ece = data["ece"]
        group = "M" if label in config.MODES_4 else "E"
        ax.set_title(f"{label} [{group}]\nECE={ece:.4f}", fontsize=9)
        ax.set_xlabel("Predicted P")
        ax.set_ylabel("Observed freq")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    for idx in range(n_labels, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Calibration (Reliability Diagrams)", fontsize=13)
    plt.tight_layout()
    fig.savefig(plot_dir / "calibration_diagrams.png")
    plt.close(fig)
    print(f"  ✓ calibration_diagrams.png")


# ═══════════════════════════════════════════════════════════════════════════
#  STRATIFICATION PLOTS  (Objective 4)
# ═══════════════════════════════════════════════════════════════════════════

def plot_density_stratification(density_results, out_dir):
    """Bar chart of error metrics by density stratum."""
    plot_dir = _ensure_dir(out_dir)

    strata = density_results.get("strata", [])
    if not strata:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Mean error by density bin
    ax = axes[0]
    bins = [s["density_bin"] for s in strata]
    means = [s["mean_error"] for s in strata]
    ns = [s["n"] for s in strata]

    bars = ax.bar(range(len(bins)), means, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(bins, ns)],
                       fontsize=9)
    ax.set_xlabel("Density bin (gold label count)")
    ax.set_ylabel("Mean Hamming Error")
    ax.set_title("Error by Label Density")

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    # Right: Exact match rate by density bin
    ax = axes[1]
    em_rates = [s.get("exact_match_rate", 0) for s in strata]
    bars = ax.bar(range(len(bins)), em_rates, color="#4CAF50", alpha=0.8)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(bins, ns)],
                       fontsize=9)
    ax.set_xlabel("Density bin")
    ax.set_ylabel("Exact Match Rate")
    ax.set_title("Exact Match by Label Density")

    # Add Spearman annotation
    sp = density_results.get("spearman_global", {})
    rho = sp.get("rho", np.nan)
    p = sp.get("p_value", np.nan)
    if not np.isnan(rho):
        fig.text(0.5, 0.01,
                 f"Spearman ρ = {rho:.3f}, p = {p:.2e}",
                 ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(plot_dir / "density_stratification.png")
    plt.close(fig)
    print(f"  ✓ density_stratification.png")


def plot_cross_stratification_heatmap(cross_results, out_dir):
    """2D heatmap: density × length → mean error."""
    plot_dir = _ensure_dir(out_dir)

    grid = cross_results.get("grid")
    grid_n = cross_results.get("grid_n")
    if grid is None or grid.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Annotation: mean (n=count)
    annot = grid.copy().astype(str)
    for r in annot.index:
        for c in annot.columns:
            m = grid.loc[r, c]
            n = grid_n.loc[r, c] if not pd.isna(grid_n.loc[r, c]) else 0
            if pd.isna(m):
                annot.loc[r, c] = ""
            else:
                annot.loc[r, c] = f"{m:.3f}\n(n={int(n)})"

    sns.heatmap(grid.astype(float), annot=annot, fmt="", cmap="RdYlGn_r",
                ax=ax, linewidths=0.5, cbar_kws={"label": "Mean Error"})
    ax.set_title("Cross-Stratification: Density × Length → Hamming Error")
    ax.set_ylabel("Label density")
    ax.set_xlabel("Word count")
    plt.tight_layout()
    fig.savefig(plot_dir / "cross_stratification.png")
    plt.close(fig)
    print(f"  ✓ cross_stratification.png")


# ═══════════════════════════════════════════════════════════════════════════
#  MASTER PLOT FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def plot_all(df, eval_labels, out_dir, *,
             label_errors_df=None,
             univar_results=None,
             bivar_results=None,
             rf_model=None, shap_values=None, X_df=None, feature_names=None,
             cond_results=None,
             interaction_results=None,
             logit_df=None,
             calibration_data=None,
             density_results=None,
             length_results=None,
             cross_results=None,
             metric="hamming_12"):
    """
    Master function that calls all individual plot functions.
    Tolerates None inputs gracefully.
    """
    print(f"\n  ═══ Generating Plots ═══")

    plot_error_distributions(df, out_dir, metric)

    if label_errors_df is not None:
        plot_per_label_errors(label_errors_df, df, eval_labels, out_dir)

    if univar_results:
        plot_univariate(univar_results, df, out_dir, metric)

    if bivar_results:
        plot_bivariate_heatmaps(bivar_results, df, out_dir, metric)

    if rf_model is not None:
        fn = feature_names or []
        plot_rf_importance(rf_model, fn, out_dir)

    if shap_values is not None and X_df is not None:
        plot_shap_summary(shap_values, X_df, out_dir)

    if cond_results:
        plot_conditional_heatmaps(cond_results, out_dir)

    if interaction_results:
        plot_interaction_matrix(interaction_results, out_dir)

    if logit_df is not None and not logit_df.empty:
        plot_logit_distributions(logit_df, df, out_dir)

    if calibration_data:
        plot_calibration_diagrams(calibration_data, out_dir)

    if density_results:
        plot_density_stratification(density_results, out_dir)

    if cross_results:
        plot_cross_stratification_heatmap(cross_results, out_dir)

    print(f"  ✓ All plots saved to {Path(out_dir) / 'plots'}")
