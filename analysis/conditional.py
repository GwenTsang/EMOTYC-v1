# -*- coding: utf-8 -*-
"""
Conditional Analysis — Mode ↔ Emotion Error Analysis & Interaction Detection
════════════════════════════════════════════════════════════════════════════════

Implements:
  - Objective 1: Conditional error analysis (which modes degrade emotion
    detection? which emotions degrade mode detection?)
  - Objective 2: Interaction & combination analysis (synergies vs. conflicts,
    error co-occurrence)
"""

import numpy as np
import pandas as pd

from . import config


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _per_label_f1(gold, pred):
    """Compute F1 for a single binary label pair."""
    tp = ((gold == 1) & (pred == 1)).sum()
    fp = ((gold == 0) & (pred == 1)).sum()
    fn = ((gold == 1) & (pred == 0)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


# ═══════════════════════════════════════════════════════════════════════════
#  OBJECTIVE 1: CONDITIONAL ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def conditional_mode_emotion_analysis(df):
    """
    Conditional error analysis between expression modes and emotions.

    For each mode m:
        - Filter to rows where gold_m=1 (the mode is present in gold)
        - Compute per-emotion F1 within this stratum
        - Compare to global (unstratified) per-emotion F1
        - Δ_F1(emotion | mode) = stratified_F1 - global_F1

    Symmetrically for each emotion e:
        - Filter to rows where gold_e=1
        - Compute per-mode F1 within this stratum

    Returns
    -------
    dict with keys:
        "delta_f1_emotion_given_mode" : pd.DataFrame
            Rows=modes, Cols=emotions, Values=Δ_F1
        "delta_f1_mode_given_emotion" : pd.DataFrame
            Rows=emotions, Cols=modes, Values=Δ_F1
        "stratified_metrics_emotion_given_mode" : dict
        "stratified_metrics_mode_given_emotion" : dict
        "global_emotion_f1" : dict
        "global_mode_f1" : dict
    """
    emotions = [e for e in config.EMOTION_12
                if e in df.columns and f"pred_{e}" in df.columns]
    modes = [m for m in config.MODES_4
             if m in df.columns and f"pred_{m}" in df.columns]

    if not emotions or not modes:
        print("  ⚠ Colonnes modes ou émotions manquantes. "
              "Analyse conditionnelle sautée.")
        return {}

    # ── Global F1 per emotion ─────────────────────────────────────────
    global_emo_f1 = {}
    for e in emotions:
        res = _per_label_f1(df[e].values.astype(int),
                            df[f"pred_{e}"].values.astype(int))
        global_emo_f1[e] = res["f1"]

    # ── Global F1 per mode ────────────────────────────────────────────
    global_mode_f1 = {}
    for m in modes:
        res = _per_label_f1(df[m].values.astype(int),
                            df[f"pred_{m}"].values.astype(int))
        global_mode_f1[m] = res["f1"]

    # ── Stratify by mode → emotion metrics ────────────────────────────
    delta_emo_given_mode = pd.DataFrame(
        index=modes, columns=emotions, dtype=float
    )
    strat_emo_given_mode = {}

    for m in modes:
        mask = df[m].values.astype(int) == 1
        n_stratum = mask.sum()
        if n_stratum < 5:
            continue

        strat_emo_given_mode[m] = {"n": n_stratum, "per_emotion": {}}
        df_stratum = df.loc[mask]

        for e in emotions:
            res = _per_label_f1(
                df_stratum[e].values.astype(int),
                df_stratum[f"pred_{e}"].values.astype(int),
            )
            delta = res["f1"] - global_emo_f1[e]
            delta_emo_given_mode.loc[m, e] = delta
            strat_emo_given_mode[m]["per_emotion"][e] = {
                **res, "delta_f1": delta, "global_f1": global_emo_f1[e]
            }

    # ── Stratify by emotion → mode metrics ────────────────────────────
    delta_mode_given_emo = pd.DataFrame(
        index=emotions, columns=modes, dtype=float
    )
    strat_mode_given_emo = {}

    for e in emotions:
        mask = df[e].values.astype(int) == 1
        n_stratum = mask.sum()
        if n_stratum < 5:
            continue

        strat_mode_given_emo[e] = {"n": n_stratum, "per_mode": {}}
        df_stratum = df.loc[mask]

        for m in modes:
            res = _per_label_f1(
                df_stratum[m].values.astype(int),
                df_stratum[f"pred_{m}"].values.astype(int),
            )
            delta = res["f1"] - global_mode_f1[m]
            delta_mode_given_emo.loc[e, m] = delta
            strat_mode_given_emo[e]["per_mode"][m] = {
                **res, "delta_f1": delta, "global_f1": global_mode_f1[m]
            }

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n  ═══ Conditional Analysis: Emotion F1 | Mode Present ═══")
    for m in modes:
        if m in strat_emo_given_mode:
            n = strat_emo_given_mode[m]["n"]
            print(f"\n  Mode={m} (n={n}):")
            for e in emotions:
                if e in strat_emo_given_mode[m]["per_emotion"]:
                    info = strat_emo_given_mode[m]["per_emotion"][e]
                    sign = "+" if info["delta_f1"] >= 0 else ""
                    print(f"    {e:<15s}  F1={info['f1']:.3f}  "
                          f"(global={info['global_f1']:.3f}, "
                          f"Δ={sign}{info['delta_f1']:.3f})")

    print(f"\n  ═══ Conditional Analysis: Mode F1 | Emotion Present ═══")
    for e in emotions:
        if e in strat_mode_given_emo:
            n = strat_mode_given_emo[e]["n"]
            # Only print emotions with enough samples
            if n >= 10:
                print(f"\n  Emotion={e} (n={n}):")
                for m in modes:
                    if m in strat_mode_given_emo[e]["per_mode"]:
                        info = strat_mode_given_emo[e]["per_mode"][m]
                        sign = "+" if info["delta_f1"] >= 0 else ""
                        print(f"    {m:<20s}  F1={info['f1']:.3f}  "
                              f"(global={info['global_f1']:.3f}, "
                              f"Δ={sign}{info['delta_f1']:.3f})")

    return {
        "delta_f1_emotion_given_mode": delta_emo_given_mode,
        "delta_f1_mode_given_emotion": delta_mode_given_emo,
        "stratified_metrics_emotion_given_mode": strat_emo_given_mode,
        "stratified_metrics_mode_given_emotion": strat_mode_given_emo,
        "global_emotion_f1": global_emo_f1,
        "global_mode_f1": global_mode_f1,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  OBJECTIVE 2: INTERACTION & COMBINATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def interaction_analysis(df, metric="hamming_12"):
    """
    Detects true interaction effects between (emotion, mode) pairs.

    For each (emotion e, mode m):
        - observed = E[error | gold_e=1, gold_m=1]
        - expected = E[error | gold_e=1] + E[error | gold_m=1] - E[error]
        - interaction_effect = observed - expected
        - Positive = conflict (worse than expected)
        - Negative = synergy (better than expected)

    Also computes an error co-occurrence matrix: for each pair of labels
    (L1, L2), how often do errors on L1 and L2 co-occur?

    Parameters
    ----------
    df : pd.DataFrame
        Must contain gold columns, pred columns, and the metric column.
    metric : str
        Error metric column to use. Default: "hamming_12".

    Returns
    -------
    dict with keys:
        "interaction_effects" : pd.DataFrame
            Rows=emotions, Cols=modes, Values=interaction effect
        "interaction_details" : list[dict]
            Sorted by |interaction_effect|
        "error_cooccurrence" : pd.DataFrame
            Symmetric matrix of error co-occurrence rates
    """
    if metric not in df.columns:
        print(f"  ⚠ Métrique '{metric}' introuvable. Interaction analysis sautée.")
        return {}

    emotions = [e for e in config.EMOTION_12 if e in df.columns]
    modes = [m for m in config.MODES_4 if m in df.columns]

    if not emotions or not modes:
        print("  ⚠ Colonnes manquantes. Interaction analysis sautée.")
        return {}

    global_err = df[metric].mean()

    # ── Interaction effects ───────────────────────────────────────────
    interaction_mat = pd.DataFrame(
        index=emotions, columns=modes, dtype=float
    )
    interaction_details = []

    for e in emotions:
        mask_e = df[e].values.astype(int) == 1
        err_e = df.loc[mask_e, metric].mean() if mask_e.sum() > 0 else global_err

        for m in modes:
            mask_m = df[m].values.astype(int) == 1
            mask_both = mask_e & mask_m
            n_both = mask_both.sum()

            if n_both < 3:
                continue

            err_m = df.loc[mask_m, metric].mean() if mask_m.sum() > 0 else global_err
            observed = df.loc[mask_both, metric].mean()

            # Expected under additivity (no interaction)
            expected = err_e + err_m - global_err

            interaction_effect = observed - expected

            interaction_mat.loc[e, m] = interaction_effect

            interaction_details.append({
                "emotion": e,
                "mode": m,
                "n_both": n_both,
                "observed_error": round(observed, 4),
                "expected_error": round(expected, 4),
                "interaction_effect": round(interaction_effect, 4),
                "err_e": round(err_e, 4),
                "err_m": round(err_m, 4),
                "global_err": round(global_err, 4),
                "type": "conflict" if interaction_effect > 0 else "synergy",
            })

    interaction_details.sort(key=lambda x: abs(x["interaction_effect"]), reverse=True)

    # ── Error co-occurrence matrix ────────────────────────────────────
    all_labels = emotions + modes
    err_cols = [f"err_{l}" for l in all_labels if f"err_{l}" in df.columns]
    labels_with_errs = [l for l in all_labels if f"err_{l}" in df.columns]

    n = len(df)
    cooc_mat = pd.DataFrame(
        np.zeros((len(labels_with_errs), len(labels_with_errs))),
        index=labels_with_errs, columns=labels_with_errs
    )

    for i, l1 in enumerate(labels_with_errs):
        err1 = df[f"err_{l1}"] != "OK"
        for j, l2 in enumerate(labels_with_errs):
            if j < i:
                cooc_mat.iloc[i, j] = cooc_mat.iloc[j, i]
                continue
            err2 = df[f"err_{l2}"] != "OK"
            cooc = (err1 & err2).sum()
            cooc_mat.iloc[i, j] = round(cooc / n, 4)

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n  ═══ Interaction Effects (Emotion × Mode) ═══")
    print(f"  Global error: {global_err:.4f}")
    print(f"\n  Top interactions (|effect| descending):")
    for d in interaction_details[:15]:
        sign = "+" if d["interaction_effect"] >= 0 else ""
        label = "CONFLICT" if d["type"] == "conflict" else "SYNERGY"
        print(f"    {d['emotion']:<15s} × {d['mode']:<18s}  "
              f"obs={d['observed_error']:.3f}  exp={d['expected_error']:.3f}  "
              f"Δ={sign}{d['interaction_effect']:.3f}  n={d['n_both']}  "
              f"[{label}]")

    # Top error co-occurrences (off-diagonal)
    print(f"\n  ═══ Error Co-occurrence (top pairs) ═══")
    cooc_pairs = []
    for i, l1 in enumerate(labels_with_errs):
        for j, l2 in enumerate(labels_with_errs):
            if j <= i:
                continue
            cooc_pairs.append((l1, l2, cooc_mat.iloc[i, j]))
    cooc_pairs.sort(key=lambda x: x[2], reverse=True)
    for l1, l2, val in cooc_pairs[:10]:
        print(f"    {l1:<15s} & {l2:<18s}  cooc_rate={val:.4f}")

    return {
        "interaction_effects": interaction_mat,
        "interaction_details": interaction_details,
        "error_cooccurrence": cooc_mat,
    }


def combination_profile_analysis(df, metric="hamming_12"):
    """
    Groups samples by their (emotion vector, mode vector) configuration
    and identifies the highest/lowest error profiles.

    Returns
    -------
    pd.DataFrame
        Each row is a unique (emotion, mode) profile with mean error,
        count, and the active labels.
    """
    if metric not in df.columns:
        return pd.DataFrame()

    emotions = [e for e in config.EMOTION_12 if e in df.columns]
    modes = [m for m in config.MODES_4 if m in df.columns]
    all_labels = emotions + modes

    if not all_labels:
        return pd.DataFrame()

    # Create a string key for each gold profile
    df["_profile_key"] = df[all_labels].apply(
        lambda row: "|".join(str(int(v)) for v in row), axis=1
    )

    profile_stats = (
        df.groupby("_profile_key")
        .agg(
            n=("_profile_key", "size"),
            mean_error=(metric, "mean"),
            median_error=(metric, "median"),
            std_error=(metric, "std"),
        )
        .reset_index()
    )

    # Decode profile key to active labels
    def _decode_profile(key):
        vals = [int(v) for v in key.split("|")]
        active = [l for l, v in zip(all_labels, vals) if v == 1]
        return active

    profile_stats["active_labels"] = profile_stats["_profile_key"].apply(_decode_profile)
    profile_stats["density"] = profile_stats["active_labels"].apply(len)

    # Sort by error (descending)
    profile_stats = profile_stats.sort_values("mean_error", ascending=False)

    # Cleanup
    df.drop(columns=["_profile_key"], inplace=True)

    # Print top and bottom profiles
    print(f"\n  ═══ Combination Profiles (Emotion+Mode) ═══")
    print(f"  {len(profile_stats)} unique profiles")

    print(f"\n  Top 10 highest-error profiles:")
    for _, row in profile_stats.head(10).iterrows():
        labels = ", ".join(row["active_labels"]) if row["active_labels"] else "∅"
        print(f"    n={row['n']:>3d}  mean_err={row['mean_error']:.3f}  "
              f"density={row['density']}  [{labels}]")

    min_n = 5
    low_err = profile_stats[profile_stats["n"] >= min_n].tail(10)
    if len(low_err) > 0:
        print(f"\n  Top 10 lowest-error profiles (n≥{min_n}):")
        for _, row in low_err.iterrows():
            labels = ", ".join(row["active_labels"]) if row["active_labels"] else "∅"
            print(f"    n={row['n']:>3d}  mean_err={row['mean_error']:.3f}  "
                  f"density={row['density']}  [{labels}]")

    return profile_stats
