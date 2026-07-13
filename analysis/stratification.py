# -*- coding: utf-8 -*-
"""
Stratification — Density and Length Stratification
══════════════════════════════════════════════════

Implements Objective 4:
  - Density-stratified evaluation (label density quartiles → performance)
  - Text length stratification (word count bins → performance)
  - Cross-stratification (density × length 2D grid)
"""

import numpy as np
import pandas as pd
from scipy import stats

from . import config


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _compute_stratum_metrics(df_stratum, eval_labels, metric_col="hamming_12"):
    """Compute summary metrics for a stratum."""
    n = len(df_stratum)
    if n == 0:
        return {"n": 0}

    result = {
        "n": n,
        "mean_error": round(df_stratum[metric_col].mean(), 4),
        "median_error": round(df_stratum[metric_col].median(), 4),
        "std_error": round(df_stratum[metric_col].std(), 4),
        "exact_match_rate": round(
            (df_stratum[metric_col] == 0).mean(), 4
        ),
    }

    # Per-label F1 within stratum
    per_label_f1 = {}
    for label in eval_labels:
        if label in df_stratum.columns and f"pred_{label}" in df_stratum.columns:
            g = df_stratum[label].values.astype(int)
            p = df_stratum[f"pred_{label}"].values.astype(int)
            tp = ((g == 1) & (p == 1)).sum()
            fp = ((g == 0) & (p == 1)).sum()
            fn = ((g == 1) & (p == 0)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_label_f1[label] = round(f1, 4)
    result["per_label_f1"] = per_label_f1

    # Annotation violation rate (emotion without mode)
    if "emotion_density_12" in df_stratum.columns:
        emotions = [e for e in config.EMOTION_12 if e in df_stratum.columns]
        modes = [m for m in config.MODES_4 if f"pred_{m}" in df_stratum.columns]
        if emotions and modes:
            any_emo = df_stratum[emotions].sum(axis=1) > 0
            pred_mode_cols = [f"pred_{m}" for m in modes]
            any_mode = df_stratum[pred_mode_cols].sum(axis=1) > 0
            viol = (any_emo & ~any_mode).sum()
            result["emotion_no_mode_rate"] = round(viol / n, 4)

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  DENSITY-STRATIFIED ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def density_stratified_analysis(df, metric="hamming_12"):
    """
    Bins samples by gold label density (quartiles) and computes
    per-stratum performance metrics.

    Also computes Spearman rank correlation between density and error.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'emotion_density_12' and the metric column.
    metric : str
        Error metric column.

    Returns
    -------
    dict with keys:
        "strata" : list[dict]
            Per-stratum metrics.
        "spearman_global" : dict
            Spearman correlation (rho, p-value) globally.
    """
    density_col = "emotion_density_12"
    if density_col not in df.columns:
        print("  ⚠ 'emotion_density_12' non trouvée. Lancez add_density_features d'abord.")
        return {}

    if metric not in df.columns:
        print(f"  ⚠ Métrique '{metric}' non trouvée.")
        return {}

    eval_labels = [e for e in config.EMOTION_12
                   if e in df.columns and f"pred_{e}" in df.columns]

    # ── Quartile binning ──────────────────────────────────────────────
    density_vals = df[density_col].values
    # Use fixed bins based on observed density values
    unique_densities = sorted(df[density_col].unique())
    if len(unique_densities) <= 4:
        # Few unique values: use each as a bin
        bins = unique_densities
        df["_density_bin"] = df[density_col].astype(int)
    else:
        # Quartile-based
        try:
            df["_density_bin"] = pd.qcut(
                df[density_col], q=4, labels=["Q1_sparse", "Q2", "Q3", "Q4_dense"],
                duplicates="drop"
            )
        except ValueError:
            # Fallback: use fixed integer bins
            df["_density_bin"] = pd.cut(
                df[density_col],
                bins=[-0.5, 0.5, 1.5, 2.5, max(density_vals) + 0.5],
                labels=["0", "1", "2", "3+"]
            )

    strata = []
    for bin_label in sorted(df["_density_bin"].unique()):
        mask = df["_density_bin"] == bin_label
        stratum_metrics = _compute_stratum_metrics(
            df.loc[mask], eval_labels, metric
        )
        stratum_metrics["density_bin"] = str(bin_label)
        stratum_metrics["density_range"] = (
            f"[{df.loc[mask, density_col].min():.0f}, "
            f"{df.loc[mask, density_col].max():.0f}]"
        )
        strata.append(stratum_metrics)

    # ── Spearman correlation ──────────────────────────────────────────
    valid = df[[density_col, metric]].dropna()
    if len(valid) > 5:
        rho, p_val = stats.spearmanr(valid[density_col], valid[metric])
        spearman_global = {"rho": round(rho, 4), "p_value": p_val}
    else:
        spearman_global = {"rho": np.nan, "p_value": np.nan}

    # Cleanup
    df.drop(columns=["_density_bin"], inplace=True, errors="ignore")

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n  ═══ Density-Stratified Analysis ═══")
    print(f"  Spearman(density, {metric}): "
          f"ρ={spearman_global['rho']:.3f}, "
          f"p={spearman_global['p_value']:.2e}")

    for s in strata:
        print(f"\n  Density bin {s['density_bin']} {s['density_range']} "
              f"(n={s['n']}):")
        print(f"    mean_err={s['mean_error']:.4f}  "
              f"median={s['median_error']:.4f}  "
              f"exact_match={s['exact_match_rate']:.3f}")

    return {
        "strata": strata,
        "spearman_global": spearman_global,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  LENGTH-STRATIFIED ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def length_stratified_analysis(df, metric="hamming_12"):
    """
    Bins samples by word count and computes per-stratum performance.
    Tests for a monotonic relationship using Jonckheere-Terpstra.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'word_count' and the metric column.
    metric : str

    Returns
    -------
    dict with keys:
        "strata" : list[dict]
        "spearman" : dict
        "trend_test" : dict  (Jonckheere-Terpstra approximation)
    """
    if "word_count" not in df.columns or metric not in df.columns:
        print("  ⚠ Colonnes manquantes pour l'analyse par longueur.")
        return {}

    eval_labels = [e for e in config.EMOTION_12
                   if e in df.columns and f"pred_{e}" in df.columns]

    # ── Tercile binning ───────────────────────────────────────────────
    try:
        df["_len_bin"] = pd.qcut(
            df["word_count"], q=3,
            labels=["short", "medium", "long"],
            duplicates="drop"
        )
    except ValueError:
        q33, q66 = df["word_count"].quantile([0.33, 0.66])
        df["_len_bin"] = pd.cut(
            df["word_count"],
            bins=[-1, q33, q66, df["word_count"].max() + 1],
            labels=["short", "medium", "long"]
        )

    strata = []
    group_arrays = []
    for bin_label in ["short", "medium", "long"]:
        mask = df["_len_bin"] == bin_label
        if mask.sum() == 0:
            continue
        stratum_metrics = _compute_stratum_metrics(
            df.loc[mask], eval_labels, metric
        )
        stratum_metrics["length_bin"] = bin_label
        stratum_metrics["word_count_range"] = (
            f"[{df.loc[mask, 'word_count'].min():.0f}, "
            f"{df.loc[mask, 'word_count'].max():.0f}]"
        )
        strata.append(stratum_metrics)
        group_arrays.append(df.loc[mask, metric].values)

    # ── Spearman ──────────────────────────────────────────────────────
    valid = df[["word_count", metric]].dropna()
    if len(valid) > 5:
        rho, p_val = stats.spearmanr(valid["word_count"], valid[metric])
        spearman = {"rho": round(rho, 4), "p_value": p_val}
    else:
        spearman = {"rho": np.nan, "p_value": np.nan}

    # ── Length-group trend summary ──────────────────────────────────────
    trend_test = {}
    if len(group_arrays) >= 3:
        kw_stat, kw_p = stats.kruskal(*[g for g in group_arrays if len(g) > 0])
        trend_test = {
            "kruskal_stat": round(kw_stat, 4),
            "kruskal_p": kw_p,
            "direction": (
                "increasing" if spearman["rho"] > 0 else "decreasing"
            ) if not np.isnan(spearman.get("rho", np.nan)) else "unknown",
        }

    # Cleanup
    df.drop(columns=["_len_bin"], inplace=True, errors="ignore")

    # ── Print ─────────────────────────────────────────────────────────
    print(f"\n  ═══ Length-Stratified Analysis ═══")
    print(f"  Spearman(word_count, {metric}): "
          f"ρ={spearman['rho']:.3f}, p={spearman['p_value']:.2e}")
    if trend_test:
        print(f"  Kruskal-Wallis: H={trend_test['kruskal_stat']:.2f}, "
              f"p={trend_test['kruskal_p']:.2e}")
        print(f"  Trend direction: {trend_test['direction']}")

    for s in strata:
        print(f"\n  Length bin '{s['length_bin']}' {s['word_count_range']} "
              f"(n={s['n']}):")
        print(f"    mean_err={s['mean_error']:.4f}  "
              f"median={s['median_error']:.4f}  "
              f"exact_match={s['exact_match_rate']:.3f}")

    return {
        "strata": strata,
        "spearman": spearman,
        "trend_test": trend_test,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  CROSS-STRATIFICATION (Density × Length)
# ═══════════════════════════════════════════════════════════════════════════

def cross_stratification(df, metric="hamming_12"):
    """
    2D grid of performance: density bins × length bins.
    Identifies the "danger zone" combinations.

    Returns
    -------
    dict with keys:
        "grid" : pd.DataFrame (pivot table of mean errors)
        "grid_n" : pd.DataFrame (pivot table of counts)
        "danger_zones" : list[dict] (combinations with highest error)
    """
    density_col = "emotion_density_12"
    if density_col not in df.columns or "word_count" not in df.columns:
        return {}
    if metric not in df.columns:
        return {}

    # Create bins
    try:
        df["_d_bin"] = pd.qcut(df[density_col], q=3,
                                labels=["sparse", "medium", "dense"],
                                duplicates="drop")
    except ValueError:
        df["_d_bin"] = pd.cut(
            df[density_col],
            bins=[-0.5, 0.5, 1.5, max(df[density_col]) + 0.5],
            labels=["0", "1", "2+"]
        )

    try:
        df["_l_bin"] = pd.qcut(df["word_count"], q=3,
                                labels=["short", "medium", "long"],
                                duplicates="drop")
    except ValueError:
        q33, q66 = df["word_count"].quantile([0.33, 0.66])
        df["_l_bin"] = pd.cut(
            df["word_count"],
            bins=[-1, q33, q66, df["word_count"].max() + 1],
            labels=["short", "medium", "long"]
        )

    # Pivot tables
    grid_mean = df.pivot_table(
        values=metric, index="_d_bin", columns="_l_bin",
        aggfunc="mean"
    )
    grid_n = df.pivot_table(
        values=metric, index="_d_bin", columns="_l_bin",
        aggfunc="count"
    )

    # Danger zones: top cells by mean error (with n >= 5)
    danger_zones = []
    for d_bin in grid_mean.index:
        for l_bin in grid_mean.columns:
            err = grid_mean.loc[d_bin, l_bin]
            n = grid_n.loc[d_bin, l_bin] if not pd.isna(grid_n.loc[d_bin, l_bin]) else 0
            if pd.notna(err) and n >= 5:
                danger_zones.append({
                    "density_bin": str(d_bin),
                    "length_bin": str(l_bin),
                    "mean_error": round(err, 4),
                    "n": int(n),
                })
    danger_zones.sort(key=lambda x: x["mean_error"], reverse=True)

    # Cleanup
    df.drop(columns=["_d_bin", "_l_bin"], inplace=True, errors="ignore")

    # Print
    print(f"\n  ═══ Cross-Stratification (Density × Length) ═══")
    print(f"\n  Mean {metric}:")
    print(grid_mean.round(4).to_string())
    print(f"\n  Counts:")
    print(grid_n.to_string())

    if danger_zones:
        print(f"\n  Danger zones (highest error):")
        for dz in danger_zones[:5]:
            print(f"    density={dz['density_bin']}, length={dz['length_bin']}  "
                  f"mean_err={dz['mean_error']:.4f}  n={dz['n']}")

    return {
        "grid": grid_mean,
        "grid_n": grid_n,
        "danger_zones": danger_zones,
    }
