# -*- coding: utf-8 -*-
"""
Metrics — Error Metrics, Brier Scores, Annotation Violations
══════════════════════════════════════════════════════════════

Computes:
  - Sample-level error metrics (Hamming, Jaccard, weighted Hamming)
  - Per-label error decomposition (FP/FN)
  - Brier score with reliability/resolution/uncertainty decomposition
  - Annotation scheme violation detection
"""

import numpy as np
import pandas as pd

from . import config


# ═══════════════════════════════════════════════════════════════════════════
#  CORE ERROR METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_error_metrics(df, label_group="emotions_12"):
    """
    Calcule les métriques d'erreur par échantillon.

    Metrics added to df:
        n_errors    — number of mismatched labels
        hamming     — n_errors / K
        jaccard_error — 1 - Jaccard(gold, pred), J(∅,∅)=1
        weighted_hamming — inverse-prevalence-weighted Hamming
        err_{label} — per-label error type ("OK", "FP", "FN")
        error_category — discretized error level

    Parameters
    ----------
    df : pd.DataFrame
        Must contain gold columns and pred_* columns.
    label_group : str
        Key in config.ANNOTATION_GROUPS. Default: "emotions_12".

    Returns
    -------
    df : pd.DataFrame
        Augmented with error columns.
    eval_labels : list[str]
        Labels actually evaluated (intersection of group and available columns).
    """
    group_labels = config.ANNOTATION_GROUPS.get(label_group, config.EMOTION_12)

    # Find labels that have both gold and pred columns
    eval_labels = [
        l for l in group_labels
        if l in df.columns and f"pred_{l}" in df.columns
    ]

    if not eval_labels:
        raise ValueError(
            f"Aucune prédiction trouvée pour le groupe '{label_group}'. "
            "Lancez l'inférence d'abord."
        )

    K = len(eval_labels)
    suffix = label_group.split("_")[-1] if "_" in label_group else label_group

    gold_mat = df[eval_labels].values.astype(int)
    pred_mat = df[[f"pred_{l}" for l in eval_labels]].values.astype(int)

    # 1. Raw error count
    errors = np.abs(gold_mat - pred_mat)
    df[f"n_errors_{suffix}"] = errors.sum(axis=1)

    # 2. Hamming error (normalized)
    df[f"hamming_{suffix}"] = df[f"n_errors_{suffix}"] / K

    # 3. Jaccard error
    intersection = (gold_mat & pred_mat).sum(axis=1).astype(float)
    union = (gold_mat | pred_mat).sum(axis=1).astype(float)
    jaccard_score = np.ones_like(union)  # J(∅,∅)=1
    np.divide(intersection, union, out=jaccard_score, where=union > 0)
    df[f"jaccard_error_{suffix}"] = 1.0 - jaccard_score

    # 4. Prevalence-weighted Hamming
    prevalences = gold_mat.mean(axis=0)
    weights = 1.0 / np.maximum(prevalences, 0.01)
    weights = weights / weights.sum()
    df[f"weighted_hamming_{suffix}"] = (errors * weights[np.newaxis, :]).sum(axis=1)

    # 5. Per-label error decomposition
    err_cols = {}
    for j, label in enumerate(eval_labels):
        g = gold_mat[:, j]
        p = pred_mat[:, j]
        err_cols[f"err_{label}"] = np.where(
            g == p, "OK",
            np.where(p > g, "FP", "FN")
        )
    df = pd.concat([df, pd.DataFrame(err_cols, index=df.index)], axis=1)

    # 6. Error category
    df["error_category"] = pd.cut(
        df[f"hamming_{suffix}"],
        bins=[-0.001, 0, 0.1, 0.3, 1.0],
        labels=["exact_match", "low_error", "medium_error", "high_error"],
    )

    # ── Backward compatibility aliases ────────────────────────────────
    if suffix == "12":
        for alias_from, alias_to in [
            ("n_errors_12", "n_errors_12"),
            ("hamming_12", "hamming_12"),
            ("jaccard_error_12", "jaccard_error_12"),
            ("weighted_hamming_12", "weighted_hamming_12"),
        ]:
            pass  # already named correctly

    # Summary
    h_col = f"hamming_{suffix}"
    n_col = f"n_errors_{suffix}"
    j_col = f"jaccard_error_{suffix}"
    w_col = f"weighted_hamming_{suffix}"

    print(f"\n  ═══ Résumé des erreurs ({label_group}, K={K}) ═══")
    print(f"  Hamming moyen        : {df[h_col].mean():.4f}")
    print(f"  Hamming médian       : {df[h_col].median():.4f}")
    print(f"  Jaccard error moyen  : {df[j_col].mean():.4f}")
    print(f"  Exact match rate     : {(df[n_col] == 0).mean():.4f}")
    print(f"  Weighted Hamming moy : {df[w_col].mean():.4f}")
    print(f"  Distribution n_errors:")
    for ne in sorted(df[n_col].unique()):
        pct = (df[n_col] == ne).mean() * 100
        print(f"    {ne} erreurs: {pct:5.1f}%")

    return df, eval_labels


# ═══════════════════════════════════════════════════════════════════════════
#  PER-LABEL ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def compute_per_label_errors(df, eval_labels):
    """
    Analyse les taux de FP/FN par label.

    Returns
    -------
    pd.DataFrame
        One row per label with FP_rate, FN_rate, accuracy, prevalence.
    """
    label_errors = []
    for label in eval_labels:
        err_col = f"err_{label}"
        if err_col not in df.columns:
            continue
        n = len(df)
        n_fp = (df[err_col] == "FP").sum()
        n_fn = (df[err_col] == "FN").sum()
        n_ok = (df[err_col] == "OK").sum()
        prev = df[label].mean() if label in df.columns else np.nan
        label_errors.append({
            "label": label,
            "FP_rate": round(n_fp / n, 4),
            "FN_rate": round(n_fn / n, 4),
            "accuracy": round(n_ok / n, 4),
            "prevalence": round(prev, 4),
            "n_FP": n_fp,
            "n_FN": n_fn,
        })
    return pd.DataFrame(label_errors)


# ═══════════════════════════════════════════════════════════════════════════
#  BRIER SCORE DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════

def compute_brier_scores(df, label_list, n_bins=10):
    """
    Computes Brier score with reliability/resolution/uncertainty decomposition
    for each label.

    The Brier score decomposes as:
        BS = reliability - resolution + uncertainty

    Where:
        - reliability: calibration error (lower is better)
        - resolution:  discriminative ability (higher is better)
        - uncertainty: inherent data uncertainty (constant for fixed labels)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain proba_{label} and gold {label} columns.
    label_list : list[str]
        Labels to evaluate.
    n_bins : int
        Number of bins for the decomposition.

    Returns
    -------
    pd.DataFrame
        Columns: label, brier_score, reliability, resolution, uncertainty,
                 calibration_error (ECE)
    """
    results = []
    for label in label_list:
        proba_col = f"proba_{label}"
        if proba_col not in df.columns or label not in df.columns:
            continue

        probas = df[proba_col].values.astype(float)
        golds = df[label].values.astype(float)

        # Overall Brier score
        brier = np.mean((probas - golds) ** 2)

        # Decomposition via binning
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probas, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        N = len(probas)
        o_bar = golds.mean()  # overall positive rate

        reliability = 0.0
        resolution = 0.0
        uncertainty = o_bar * (1 - o_bar)
        ece = 0.0  # Expected Calibration Error

        for k in range(n_bins):
            mask = bin_indices == k
            n_k = mask.sum()
            if n_k == 0:
                continue

            o_k = golds[mask].mean()      # observed frequency in bin
            f_k = probas[mask].mean()      # mean predicted probability in bin

            reliability += n_k * (f_k - o_k) ** 2
            resolution += n_k * (o_k - o_bar) ** 2
            ece += n_k * abs(f_k - o_k)

        reliability /= N
        resolution /= N
        ece /= N

        results.append({
            "label": label,
            "brier_score": round(brier, 6),
            "reliability": round(reliability, 6),
            "resolution": round(resolution, 6),
            "uncertainty": round(uncertainty, 6),
            "ece": round(ece, 6),
            "n_samples": N,
            "prevalence": round(o_bar, 4),
        })

    brier_df = pd.DataFrame(results)

    if len(brier_df) > 0:
        print(f"\n  ═══ Brier Score Decomposition ({len(brier_df)} labels) ═══")
        print(f"  {'Label':<15s} {'BS':>8s} {'Reliab':>8s} {'Resol':>8s} "
              f"{'Uncert':>8s} {'ECE':>8s}")
        for _, r in brier_df.iterrows():
            print(f"  {r['label']:<15s} {r['brier_score']:>8.4f} "
                  f"{r['reliability']:>8.4f} {r['resolution']:>8.4f} "
                  f"{r['uncertainty']:>8.4f} {r['ece']:>8.4f}")

    return brier_df


# ═══════════════════════════════════════════════════════════════════════════
#  ANNOTATION SCHEME VIOLATION DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def compute_annotation_violations(df):
    """
    Detects annotation scheme violations in predictions.

    Violation types (from sanity_checks.md):
        1. emo_no_emotion:    Emo=1 but no emotion active
        2. emotion_no_emo:    Emotion active but Emo=0
        3. base_no_basic:     Base=1 but no basic emotion active
        4. basic_no_base:     Basic emotion active but Base=0
        5. complex_no_cpx:    Complexe=1 but no complex emotion active
        6. cpx_no_complex:    Complex emotion active but Complexe=0
        7. mode_no_emotion:   Mode active but no emotion (M>0, E=0)
        8. emotion_no_mode:   Emotion active but no mode (E>0, M=0)

    Returns
    -------
    pd.DataFrame
        Per-sample violation indicators + summary.
    """
    violations = pd.DataFrame(index=df.index)
    pred_prefix = "pred_"

    # Helper: check if pred column exists and get its values
    def _pred(label):
        col = f"{pred_prefix}{label}"
        if col in df.columns:
            return df[col].fillna(0).astype(int).values
        return np.zeros(len(df), dtype=int)

    # Predicted emotion activity
    pred_emotions = np.column_stack([_pred(e) for e in config.EMOTION_12])
    any_emotion = pred_emotions.sum(axis=1) > 0

    # Predicted basic / complex emotion activity
    pred_basic = np.column_stack([_pred(e) for e in config.BASIC_EMOTIONS])
    any_basic = pred_basic.sum(axis=1) > 0

    pred_complex = np.column_stack([_pred(e) for e in config.COMPLEX_EMOTIONS])
    any_complex = pred_complex.sum(axis=1) > 0

    # Predicted mode activity
    pred_modes = np.column_stack([_pred(m) for m in config.MODES_4])
    any_mode = pred_modes.sum(axis=1) > 0

    # Emo, Base, Complexe
    pred_emo = _pred("Emo")
    pred_base = _pred("Base")
    pred_complexe = _pred("Complexe")

    # ── Violation checks ──────────────────────────────────────────────
    violations["emo_no_emotion"] = ((pred_emo == 1) & ~any_emotion).astype(int)
    violations["emotion_no_emo"] = (any_emotion & (pred_emo == 0)).astype(int)
    violations["base_no_basic"] = ((pred_base == 1) & ~any_basic).astype(int)
    violations["basic_no_base"] = (any_basic & (pred_base == 0)).astype(int)
    violations["complex_no_cpx"] = ((pred_complexe == 1) & ~any_complex).astype(int)
    violations["cpx_no_complex"] = (any_complex & (pred_complexe == 0)).astype(int)
    violations["mode_no_emotion"] = (any_mode & ~any_emotion).astype(int)
    violations["emotion_no_mode"] = (any_emotion & ~any_mode).astype(int)
    violations["any_violation"] = (violations.sum(axis=1) > 0).astype(int)
    violations["n_violations"] = violations.drop(
        columns=["any_violation"]
    ).sum(axis=1)

    # ── Summary ───────────────────────────────────────────────────────
    n = len(df)
    print(f"\n  ═══ Annotation Scheme Violations (N={n}) ═══")
    for col in violations.columns:
        if col in ("any_violation", "n_violations"):
            continue
        count = violations[col].sum()
        pct = 100 * count / n
        print(f"  {col:<25s}: {count:>4d}  ({pct:5.1f}%)")
    total = violations["any_violation"].sum()
    print(f"  {'ANY violation':<25s}: {total:>4d}  ({100*total/n:5.1f}%)")

    return violations
