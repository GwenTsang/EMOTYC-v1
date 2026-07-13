# -*- coding: utf-8 -*-
"""
Logit Analysis — Logit Distributions and Calibration
══════════════════════════════════════════════════════════════════════

Implements Objective 3:
  - Logit/probability distribution analysis per label (gold=0 vs gold=1)
  - Calibration analysis (reliability diagrams, Expected Calibration Error)
"""

import numpy as np
import pandas as pd

from . import config


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _proba_to_logit(p, eps=1e-7):
    """Convert probability to logit (log-odds), clamping to avoid ±inf."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


# ═══════════════════════════════════════════════════════════════════════════
#  LOGIT DISTRIBUTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def logit_distribution_analysis(df):
    """
    For each label, analyse the logit/probability distribution conditioned
    on gold=0 vs gold=1.

    Metrics computed per label:
        - logit_separation: mean(logit|gold=1) - mean(logit|gold=0)
        - proba_overlap: overlap coefficient between the two proba distributions
        - mean/std of probabilities for positive and negative gold classes

    Returns
    -------
    pd.DataFrame
        One row per label with distribution statistics.
    """
    all_labels = config.EMOTION_12 + config.MODES_4 + config.META_LABELS + config.TYPES_2
    results = []

    for label in all_labels:
        proba_col = f"proba_{label}"
        if proba_col not in df.columns or label not in df.columns:
            continue

        probas = df[proba_col].values.astype(float)
        golds = df[label].values.astype(int)

        mask_pos = golds == 1
        mask_neg = golds == 0

        n_pos = mask_pos.sum()
        n_neg = mask_neg.sum()

        if n_pos == 0 or n_neg == 0:
            results.append({
                "label": label,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "logit_separation": np.nan,
                "proba_mean_pos": np.nan,
                "proba_mean_neg": np.nan,
                "proba_std_pos": np.nan,
                "proba_std_neg": np.nan,
                "proba_overlap": np.nan,
            })
            continue

        # Logit analysis
        logits = _proba_to_logit(probas)
        logit_mean_pos = logits[mask_pos].mean()
        logit_mean_neg = logits[mask_neg].mean()
        logit_sep = logit_mean_pos - logit_mean_neg

        # Probability distribution statistics
        proba_mean_pos = probas[mask_pos].mean()
        proba_mean_neg = probas[mask_neg].mean()
        proba_std_pos = probas[mask_pos].std()
        proba_std_neg = probas[mask_neg].std()

        # Overlap coefficient (histogram-based)
        bins = np.linspace(0, 1, 51)
        hist_pos, _ = np.histogram(probas[mask_pos], bins=bins, density=True)
        hist_neg, _ = np.histogram(probas[mask_neg], bins=bins, density=True)
        # Normalize to probability distribution
        bin_width = bins[1] - bins[0]
        hist_pos_prob = hist_pos * bin_width
        hist_neg_prob = hist_neg * bin_width
        overlap = np.minimum(hist_pos_prob, hist_neg_prob).sum()

        results.append({
            "label": label,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "logit_separation": round(logit_sep, 4),
            "logit_mean_pos": round(logit_mean_pos, 4),
            "logit_mean_neg": round(logit_mean_neg, 4),
            "proba_mean_pos": round(proba_mean_pos, 4),
            "proba_mean_neg": round(proba_mean_neg, 4),
            "proba_std_pos": round(proba_std_pos, 4),
            "proba_std_neg": round(proba_std_neg, 4),
            "proba_overlap": round(overlap, 4),
        })

    logit_df = pd.DataFrame(results)

    if len(logit_df) > 0:
        print(f"\n  ═══ Logit Distribution Analysis ({len(logit_df)} labels) ═══")
        print(f"  {'Label':<18s} {'n+':<5s} {'n-':<5s} {'sep':>6s} "
              f"{'p̄(+)':>7s} {'p̄(-)':>7s} {'overlap':>8s}")
        for _, r in logit_df.iterrows():
            sep = f"{r['logit_separation']:+.2f}" if not np.isnan(r['logit_separation']) else "  N/A"
            print(f"  {r['label']:<18s} {r['n_pos']:<5d} {r['n_neg']:<5d} {sep:>6s} "
                  f"{r['proba_mean_pos']:>7.3f} {r['proba_mean_neg']:>7.3f} "
                  f"{r['proba_overlap']:>8.3f}")

    return logit_df


# ═══════════════════════════════════════════════════════════════════════════
#  CALIBRATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def calibration_analysis(df, label_list=None, n_bins=10):
    """
    Compute calibration data for reliability diagrams.

    For each label, bins predicted probabilities and computes the observed
    frequency of positives within each bin.

    Parameters
    ----------
    df : pd.DataFrame
    label_list : list[str], optional
        Labels to analyze. Default: EMOTION_12 + MODES_4
    n_bins : int

    Returns
    -------
    dict
        {label: {"bin_midpoints": [...], "observed_freq": [...],
                 "mean_pred": [...], "bin_count": [...], "ece": float}}
    """
    if label_list is None:
        label_list = config.EMOTION_12 + config.MODES_4

    calibration_data = {}

    for label in label_list:
        proba_col = f"proba_{label}"
        if proba_col not in df.columns or label not in df.columns:
            continue

        probas = df[proba_col].values.astype(float)
        golds = df[label].values.astype(float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probas, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        bin_midpoints = []
        observed_freq = []
        mean_pred = []
        bin_count = []
        ece = 0.0

        for k in range(n_bins):
            mask = bin_indices == k
            n_k = mask.sum()
            mid = (bin_edges[k] + bin_edges[k + 1]) / 2
            bin_midpoints.append(mid)
            bin_count.append(n_k)

            if n_k == 0:
                observed_freq.append(np.nan)
                mean_pred.append(np.nan)
                continue

            obs = golds[mask].mean()
            pred_mean = probas[mask].mean()
            observed_freq.append(obs)
            mean_pred.append(pred_mean)
            ece += n_k * abs(pred_mean - obs)

        ece /= len(probas)

        calibration_data[label] = {
            "bin_midpoints": bin_midpoints,
            "observed_freq": observed_freq,
            "mean_pred": mean_pred,
            "bin_count": bin_count,
            "ece": round(ece, 6),
        }

    # Summary
    print(f"\n  ═══ Calibration Summary ═══")
    emo_eces = []
    mode_eces = []
    for label, data in calibration_data.items():
        group = "mode" if label in config.MODES_4 else "emotion"
        print(f"  {label:<18s}  ECE={data['ece']:.4f}  [{group}]")
        if group == "mode":
            mode_eces.append(data["ece"])
        else:
            emo_eces.append(data["ece"])

    if emo_eces:
        print(f"\n  Mean ECE (emotions): {np.mean(emo_eces):.4f}")
    if mode_eces:
        print(f"  Mean ECE (modes):    {np.mean(mode_eces):.4f}")

    return calibration_data
