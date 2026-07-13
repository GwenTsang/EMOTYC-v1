# -*- coding: utf-8 -*-
"""
Report — Comprehensive Markdown Report Generation
═══════════════════════════════════════════════════

Generates a structured Markdown report synthesizing all analysis results.
The report is designed to be the primary deliverable: a self-contained
document that tells the story of EMOTYC's failure modes, structural
biases, and actionable recommendations.
"""

import datetime
import numpy as np
import pandas as pd
from pathlib import Path

from . import config


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _sig_stars(p):
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def _fmt_float(val, fmt=".4f"):
    """Format a float, handling NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:{fmt}}"


def _section(title, level=2):
    """Return a markdown section header."""
    prefix = "#" * level
    return f"\n{prefix} {title}\n"


def _format_itemset(items):
    """Format a mlxtend frozenset for compact Markdown tables."""
    try:
        values = list(items)
    except TypeError:
        values = [items]
    labels = [str(v) for v in values]
    return " ∧ ".join(sorted(labels))


# ═══════════════════════════════════════════════════════════════════════════
#  REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(out_dir, config_str="", **kwargs):
    """
    Generates a comprehensive Markdown report from all analysis results.

    Parameters
    ----------
    out_dir : str or Path
        Output directory (report + plots).
    config_str : str
        Configuration description string.
    **kwargs : dict
        All analysis results, keyed by name. Expected keys:

        Core:
            df, eval_labels

        Metrics:
            label_errors_df, violations_df, brier_df

        Conditional (Obj 1 & 2):
            cond_results, interaction_results, profile_stats

        Logit & Calibration (Obj 3):
            logit_df, calibration_data

        Stratification (Obj 4):
            density_results, length_results, cross_results

        Explainability:
            univar_results, bivar_results,
            rf_model, shap_values, feature_names, dt_model, rules

    Returns
    -------
    str
        The full report text.
    """
    out_dir = Path(out_dir)
    df = kwargs.get("df")
    eval_labels = kwargs.get("eval_labels", [])

    lines = []

    # ──────────────────────────────────────────────────────────────────
    #  HEADER
    # ──────────────────────────────────────────────────────────────────
    lines.append("# EMOTYC Error Analysis — Comprehensive Report")
    lines.append("")
    lines.append(f"> Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"> Configuration: `{config_str}`")
    lines.append("")

    if df is not None:
        n = len(df)
        lines.append(f"**Dataset**: {n} samples")
        lines.append("")
        lines.append(f"**Labels evaluated**: {len(eval_labels)} "
                      f"({', '.join(eval_labels[:6])}...)")
        lines.append("")

    # ──────────────────────────────────────────────────────────────────
    #  §1. GLOBAL ERROR METRICS
    # ──────────────────────────────────────────────────────────────────
    lines.append(_section("1. Global Error Metrics"))

    if df is not None and "hamming_12" in df.columns:
        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")
        lines.append(f"| Hamming Error (mean) | "
                      f"{df['hamming_12'].mean():.4f} |")
        lines.append(f"| Hamming Error (median) | "
                      f"{df['hamming_12'].median():.4f} |")
        if "jaccard_error_12" in df.columns:
            lines.append(f"| Jaccard Error (mean) | "
                          f"{df['jaccard_error_12'].mean():.4f} |")
        if "weighted_hamming_12" in df.columns:
            lines.append(f"| Weighted Hamming (mean) | "
                          f"{df['weighted_hamming_12'].mean():.4f} |")
        if "n_errors_12" in df.columns:
            lines.append(f"| Exact Match rate | "
                          f"{(df['n_errors_12'] == 0).mean():.4f} |")
        lines.append("")

    # ──────────────────────────────────────────────────────────────────
    #  §2. PER-LABEL ERROR DECOMPOSITION
    # ──────────────────────────────────────────────────────────────────
    label_errors_df = kwargs.get("label_errors_df")
    if label_errors_df is not None and not label_errors_df.empty:
        lines.append(_section("2. Per-Label Error Decomposition"))
        lines.append("| Label | Prevalence | FP rate | FN rate | "
                      "Accuracy | n_FP | n_FN |")
        lines.append("|-------|----------:|---------:|--------:|"
                      "---------:|-----:|-----:|")
        for _, r in label_errors_df.iterrows():
            lines.append(
                f"| {r['label']} | {r['prevalence']:.3f} | "
                f"{r['FP_rate']:.4f} | {r['FN_rate']:.4f} | "
                f"{r['accuracy']:.4f} | {r['n_FP']} | {r['n_FN']} |"
            )
        lines.append("")
        lines.append("![Per-label error decomposition](plots/"
                      "per_label_error_decomposition.png)")
        lines.append("")

    # ──────────────────────────────────────────────────────────────────
    #  §3. ANNOTATION SCHEME VIOLATIONS
    # ──────────────────────────────────────────────────────────────────
    violations_df = kwargs.get("violations_df")
    if violations_df is not None and not violations_df.empty:
        lines.append(_section("3. Annotation Scheme Violations"))
        lines.append("> [!WARNING]")
        lines.append("> These violations indicate structural "
                      "inconsistencies in the model's predictions.")
        lines.append("")

        n = len(violations_df)
        lines.append("| Violation Type | Count | Rate |")
        lines.append("|----------------|------:|-----:|")
        for col in violations_df.columns:
            if col in ("any_violation", "n_violations"):
                continue
            count = violations_df[col].sum()
            lines.append(f"| {col} | {count} | {100*count/n:.1f}% |")
        total = violations_df["any_violation"].sum()
        lines.append(f"| **ANY violation** | **{total}** | "
                      f"**{100*total/n:.1f}%** |")
        lines.append("")

        # Key insight
        emo_no_mode = violations_df.get("emotion_no_mode")
        if emo_no_mode is not None:
            rate = emo_no_mode.sum() / n * 100
            lines.append(f"> [!IMPORTANT]")
            lines.append(f"> The dominant violation is **emotion without "
                          f"mode** ({rate:.1f}%), confirming the structural "
                          f"weakness identified in previous sanity checks.")
            lines.append("")

    # ──────────────────────────────────────────────────────────────────
    #  §4. BRIER SCORE DECOMPOSITION
    # ──────────────────────────────────────────────────────────────────
    brier_df = kwargs.get("brier_df")
    if brier_df is not None and not brier_df.empty:
        lines.append(_section("4. Brier Score Decomposition"))
        lines.append("The Brier score decomposes as: "
                      "`BS = reliability − resolution + uncertainty`")
        lines.append("")
        lines.append("- **Reliability** (↓ better): calibration error")
        lines.append("- **Resolution** (↑ better): discriminative power")
        lines.append("- **Uncertainty**: inherent data entropy (fixed)")
        lines.append("")
        lines.append("| Label | Brier | Reliability | Resolution | "
                      "Uncertainty | ECE |")
        lines.append("|-------|------:|----------:|----------:|"
                      "----------:|----:|")
        for _, r in brier_df.iterrows():
            lines.append(
                f"| {r['label']} | {r['brier_score']:.4f} | "
                f"{r['reliability']:.4f} | {r['resolution']:.4f} | "
                f"{r['uncertainty']:.4f} | {r['ece']:.4f} |"
            )
        lines.append("")

        # Insight: compare emotions vs modes
        emo_rows = brier_df[~brier_df["label"].isin(config.MODES_4)]
        mode_rows = brier_df[brier_df["label"].isin(config.MODES_4)]
        if not mode_rows.empty and not emo_rows.empty:
            mean_ece_emo = emo_rows["ece"].mean()
            mean_ece_mode = mode_rows["ece"].mean()
            lines.append(f"> [!NOTE]")
            lines.append(f"> Mean ECE for emotions = {mean_ece_emo:.4f}, "
                          f"for modes = {mean_ece_mode:.4f}. "
                          f"{'Modes are worse calibrated.' if mean_ece_mode > mean_ece_emo else 'Emotions are worse calibrated.'}")
            lines.append("")

    # ──────────────────────────────────────────────────────────────────
    #  §5. CONDITIONAL ERROR ANALYSIS  (Objective 1)
    # ──────────────────────────────────────────────────────────────────
    cond_results = kwargs.get("cond_results")
    if cond_results:
        lines.append(_section("5. Conditional Error Analysis "
                              "(Modes ↔ Emotions)"))

        # Δ F1 (emotion | mode)
        delta_em = cond_results.get("delta_f1_emotion_given_mode")
        if delta_em is not None and not delta_em.empty:
            lines.append(_section("Which modes degrade emotion detection?", 3))
            lines.append("Δ F1 = F1(emotion | mode present) − F1(emotion global). "
                          "**Negative = degradation** when this mode is present.")
            lines.append("")
            lines.append("![Conditional: emotion given mode](plots/"
                          "conditional_emotion_given_mode.png)")
            lines.append("")

            # Extract key findings
            strat = cond_results.get("stratified_metrics_emotion_given_mode", {})
            worst_pairs = []
            for mode, data in strat.items():
                for emo, info in data.get("per_emotion", {}).items():
                    worst_pairs.append({
                        "mode": mode,
                        "emotion": emo,
                        "delta": info["delta_f1"],
                        "f1": info["f1"],
                        "n": data["n"],
                    })
            worst_pairs.sort(key=lambda x: x["delta"])
            if worst_pairs:
                lines.append("**Top degradations** (mode → emotion):")
                lines.append("")
                for p in worst_pairs[:8]:
                    if p["delta"] < 0:
                        lines.append(f"- `{p['mode']}` → `{p['emotion']}`: "
                                      f"Δ F1 = {p['delta']:+.3f} "
                                      f"(F1={p['f1']:.3f}, n={p['n']})")
                lines.append("")

        # Δ F1 (mode | emotion)
        delta_me = cond_results.get("delta_f1_mode_given_emotion")
        if delta_me is not None and not delta_me.empty:
            lines.append(_section("Which emotions degrade mode detection?", 3))
            lines.append("![Conditional: mode given emotion](plots/"
                          "conditional_mode_given_emotion.png)")
            lines.append("")

    # ──────────────────────────────────────────────────────────────────
    #  §6. INTERACTION ANALYSIS  (Objective 2)
    # ──────────────────────────────────────────────────────────────────
    interaction_results = kwargs.get("interaction_results")
    if interaction_results:
        lines.append(_section("6. Interaction & Combination Analysis"))

        details = interaction_results.get("interaction_details", [])
        if details:
            lines.append("**Interaction effect** = observed error − "
                          "expected error under additivity.")
            lines.append("")
            lines.append("- **Positive (conflict)**: the combination "
                          "performs *worse* than expected")
            lines.append("- **Negative (synergy)**: the combination "
                          "performs *better* than expected")
            lines.append("")
            lines.append("![Interaction matrix](plots/"
                          "interaction_emotion_mode.png)")
            lines.append("")

            # Top conflicts
            conflicts = [d for d in details if d["type"] == "conflict"]
            synergies = [d for d in details if d["type"] == "synergy"]

            if conflicts:
                lines.append(_section("Top Conflicts (worse than expected)", 3))
                lines.append("| Emotion | Mode | Observed | Expected "
                              "| Δ | n |")
                lines.append("|---------|------|--------:|---------:|"
                              "--:|--:|")
                for d in conflicts[:10]:
                    lines.append(
                        f"| {d['emotion']} | {d['mode']} | "
                        f"{d['observed_error']:.3f} | "
                        f"{d['expected_error']:.3f} | "
                        f"{d['interaction_effect']:+.3f} | {d['n_both']} |"
                    )
                lines.append("")

            if synergies:
                lines.append(_section("Top Synergies (better than expected)", 3))
                lines.append("| Emotion | Mode | Observed | Expected "
                              "| Δ | n |")
                lines.append("|---------|------|--------:|---------:|"
                              "--:|--:|")
                for d in synergies[:10]:
                    lines.append(
                        f"| {d['emotion']} | {d['mode']} | "
                        f"{d['observed_error']:.3f} | "
                        f"{d['expected_error']:.3f} | "
                        f"{d['interaction_effect']:+.3f} | {d['n_both']} |"
                    )
                lines.append("")

        # Error co-occurrence
        cooc = interaction_results.get("error_cooccurrence")
        if cooc is not None:
            lines.append("![Error co-occurrence](plots/"
                          "error_cooccurrence.png)")
            lines.append("")

    # Profile analysis
    profile_stats = kwargs.get("profile_stats")
    if profile_stats is not None and not profile_stats.empty:
        lines.append(_section("Combination Profiles", 3))
        lines.append(f"{len(profile_stats)} unique label configurations "
                      "found.")
        lines.append("")

        # Top worst profiles
        worst = profile_stats.head(5)
        lines.append("**Highest-error profiles:**")
        lines.append("")
        lines.append("| Active Labels | n | Mean Error | Density |")
        lines.append("|---------------|--:|-----------:|--------:|")
        for _, row in worst.iterrows():
            labels = ", ".join(row["active_labels"]) if row["active_labels"] else "∅"
            lines.append(
                f"| {labels} | {row['n']} | "
                f"{row['mean_error']:.3f} | {row['density']} |"
            )
        lines.append("")

    # ──────────────────────────────────────────────────────────────────
    #  §7. LOGIT & CALIBRATION ANALYSIS  (Objective 3)
    # ──────────────────────────────────────────────────────────────────
    logit_df = kwargs.get("logit_df")

    if logit_df is not None and not logit_df.empty:
        lines.append(_section("7. Logit & Calibration Analysis"))

    if logit_df is not None and not logit_df.empty:
        lines.append(_section("Logit Separation", 3))
        lines.append("Higher separation = better discriminability. "
                      "Labels with low separation cannot be reliably "
                      "classified regardless of threshold.")
        lines.append("")
        lines.append("| Label | n+ | n− | Separation | p̄(gold=1) | "
                      "p̄(gold=0) | Overlap |")
        lines.append("|-------|---:|---:|-----------:|----------:|"
                      "---------:|--------:|")
        for _, r in logit_df.iterrows():
            sep = _fmt_float(r.get("logit_separation"), "+.2f")
            lines.append(
                f"| {r['label']} | {r['n_pos']} | {r['n_neg']} | "
                f"{sep} | {_fmt_float(r.get('proba_mean_pos'), '.3f')} | "
                f"{_fmt_float(r.get('proba_mean_neg'), '.3f')} | "
                f"{_fmt_float(r.get('proba_overlap'), '.3f')} |"
            )
        lines.append("")
        lines.append("![Logit distributions](plots/logit_distributions.png)")
        lines.append("")

    # Calibration
    calibration_data = kwargs.get("calibration_data")
    if calibration_data:
        lines.append(_section("Calibration", 3))
        lines.append("![Calibration diagrams](plots/"
                      "calibration_diagrams.png)")
        lines.append("")

    # ──────────────────────────────────────────────────────────────────
    #  §8. DENSITY & LENGTH STRATIFICATION  (Objective 4)
    # ──────────────────────────────────────────────────────────────────
    density_results = kwargs.get("density_results")
    length_results = kwargs.get("length_results")
    cross_results = kwargs.get("cross_results")
    if density_results or length_results:
        lines.append(_section("8. Density & Length Stratification"))

    if density_results:
        lines.append(_section("Density-stratified performance", 3))
        strata = density_results.get("strata", [])
        sp = density_results.get("spearman_global", {})
        rho = sp.get("rho", np.nan)
        p = sp.get("p_value", np.nan)

        if not np.isnan(rho):
            sig = _sig_stars(p)
            lines.append(f"**Spearman correlation** (density vs Hamming): "
                          f"ρ = {rho:.3f}, p = {p:.2e} {sig}")
            lines.append("")

        if strata:
            lines.append("| Density Bin | Range | n | Mean Error | "
                          "Exact Match |")
            lines.append("|-------------|-------|--:|-----------:|"
                          "-----------:|")
            for s in strata:
                lines.append(
                    f"| {s['density_bin']} | {s.get('density_range', '')} | "
                    f"{s['n']} | {s['mean_error']:.4f} | "
                    f"{s.get('exact_match_rate', 0):.3f} |"
                )
            lines.append("")

        lines.append("![Density stratification](plots/"
                      "density_stratification.png)")
        lines.append("")

    if length_results:
        lines.append(_section("Length-stratified performance", 3))
        strata = length_results.get("strata", [])
        sp = length_results.get("spearman", {})
        rho = sp.get("rho", np.nan)
        p = sp.get("p_value", np.nan)

        if not np.isnan(rho):
            sig = _sig_stars(p)
            lines.append(f"**Spearman correlation** (word_count vs Hamming): "
                          f"ρ = {rho:.3f}, p = {p:.2e} {sig}")
            lines.append("")

        trend = length_results.get("trend_test", {})
        if trend:
            lines.append(f"Kruskal-Wallis: H={trend.get('kruskal_stat', 'N/A')}, "
                          f"p={trend.get('kruskal_p', 'N/A'):.2e}, "
                          f"direction={trend.get('direction', 'unknown')}")
            lines.append("")

        if strata:
            lines.append("| Length Bin | Range | n | Mean Error | "
                          "Exact Match |")
            lines.append("|-----------|-------|--:|-----------:|"
                          "-----------:|")
            for s in strata:
                lines.append(
                    f"| {s['length_bin']} | "
                    f"{s.get('word_count_range', '')} | "
                    f"{s['n']} | {s['mean_error']:.4f} | "
                    f"{s.get('exact_match_rate', 0):.3f} |"
                )
            lines.append("")

    if cross_results:
        lines.append(_section("Cross-stratification (Density × Length)", 3))
        lines.append("![Cross-stratification](plots/"
                      "cross_stratification.png)")
        lines.append("")

        danger = cross_results.get("danger_zones", [])
        if danger:
            lines.append("> [!CAUTION]")
            lines.append("> **Danger zones** — combinations with highest "
                          "error rates:")
            lines.append(">")
            for dz in danger[:3]:
                lines.append(
                    f"> - density={dz['density_bin']}, "
                    f"length={dz['length_bin']}: "
                    f"mean_error={dz['mean_error']:.4f} (n={dz['n']})"
                )
            lines.append("")

    # ──────────────────────────────────────────────────────────────────
    #  §9. FEATURE IMPORTANCE (Explainability)
    # ──────────────────────────────────────────────────────────────────
    rf_model = kwargs.get("rf_model")
    univar_results = kwargs.get("univar_results")
    bivar_results = kwargs.get("bivar_results")

    if rf_model or univar_results:
        lines.append(_section("9. Feature Importance & Explainability"))

    if univar_results:
        lines.append(_section("Univariate Analysis", 3))
        lines.append("Features ranked by statistical significance "
                      "(effect on Hamming error):")
        lines.append("")
        lines.append("| Feature | Test | p-value | η² | Top Level "
                      "| Mean Error |")
        lines.append("|---------|------|--------:|---:|-----------|"
                      "-----------:|")
        for r in univar_results[:12]:
            sig = _sig_stars(r["p_value"])
            top = r["levels"][0] if r["levels"] else {}
            lines.append(
                f"| {r['feature']} | {r['test']} | "
                f"{r['p_value']:.2e} {sig} | {r['eta_squared']:.3f} | "
                f"{top.get('level', '')} | "
                f"{top.get('mean_error', 0):.4f} |"
            )
        lines.append("")

    if rf_model is not None:
        lines.append(_section("Random Forest Regressor", 3))
        lines.append(f"- OOB R²: {rf_model.oob_score_:.4f}")
        lines.append("")
        lines.append("![RF Feature Importance](plots/"
                      "rf_feature_importance.png)")
        lines.append("")

        imp = rf_model.feature_importances_
        fn = kwargs.get("feature_names", [])
        sorted_idx = np.argsort(imp)[::-1]
        lines.append("| Rank | Feature | MDI Importance |")
        lines.append("|-----:|---------|---------------:|")
        for rank, idx in enumerate(sorted_idx[:15]):
            fname = fn[idx] if idx < len(fn) else f"feature_{idx}"
            lines.append(f"| {rank+1} | {fname} | {imp[idx]:.4f} |")
        lines.append("")

    shap_values = kwargs.get("shap_values")
    if shap_values is not None:
        lines.append(_section("SHAP Analysis", 3))
        lines.append("![SHAP Summary](plots/shap_summary.png)")
        lines.append("")
        lines.append("![SHAP Bar](plots/shap_bar.png)")
        lines.append("")

        fn = kwargs.get("feature_names", [])
        try:
            shap_arr = np.asarray(shap_values)
            if shap_arr.ndim >= 2:
                axes = tuple(range(shap_arr.ndim - 1))
                mean_abs = np.abs(shap_arr).mean(axis=axes)
                sorted_idx = np.argsort(mean_abs)[::-1]
                lines.append("| Rank | Feature | mean \\|SHAP\\| |")
                lines.append("|-----:|---------|-------------:|")
                for rank, idx in enumerate(sorted_idx[:15]):
                    fname = fn[idx] if idx < len(fn) else f"feature_{idx}"
                    lines.append(f"| {rank+1} | {fname} | {mean_abs[idx]:.4f} |")
                lines.append("")
        except Exception as exc:
            lines.append(f"_SHAP values were computed but could not be tabulated: {exc}_")
            lines.append("")

    rules = kwargs.get("rules")
    if rules is not None and hasattr(rules, "empty") and not rules.empty:
        lines.append(_section("Association Rules (High-Error Subset)", 3))
        lines.append("FP-Growth is run only on samples above the high-error "
                      "threshold. The rules below describe co-occurring "
                      "profiles inside failures; they are not causal effects "
                      "or lift against the full corpus.")
        lines.append("")

        if {"antecedents", "consequents", "support", "confidence", "lift"}.issubset(rules.columns):
            display_rows = []
            seen = set()
            for _, row in rules.sort_values("lift", ascending=False).iterrows():
                ant = _format_itemset(row["antecedents"])
                cons = _format_itemset(row["consequents"])
                if not ant or not cons:
                    continue
                key = (ant, cons)
                if key in seen:
                    continue
                seen.add(key)
                display_rows.append((ant, cons, row))
                if len(display_rows) >= 12:
                    break

            if display_rows:
                lines.append("| Antecedent | Consequent | Support | Confidence | Lift |")
                lines.append("|------------|------------|--------:|-----------:|-----:|")
                for ant, cons, row in display_rows:
                    lines.append(
                        f"| {ant} | {cons} | {row['support']:.3f} | "
                        f"{row['confidence']:.3f} | {row['lift']:.2f} |"
                    )
                lines.append("")
        elif {"itemsets", "support"}.issubset(rules.columns):
            lines.append("| Itemset | Support |")
            lines.append("|---------|--------:|")
            for _, row in rules.sort_values("support", ascending=False).head(12).iterrows():
                itemset = _format_itemset(row["itemsets"])
                if itemset:
                    lines.append(f"| {itemset} | {row['support']:.3f} |")
            lines.append("")

    # Decision tree
    dt_model = kwargs.get("dt_model")
    if dt_model is not None and hasattr(dt_model, "tree_text"):
        lines.append(_section("Decision Tree Rules (depth=4)", 3))
        lines.append("```")
        lines.append(dt_model.tree_text)
        lines.append("```")
        lines.append("")

    if bivar_results:
        lines.append(_section("Bivariate Interactions", 3))
        lines.append("| Pair | Error Range | Max | Min |")
        lines.append("|------|----------:|----|----:|")
        for r in bivar_results[:10]:
            lines.append(
                f"| {r['f1']} × {r['f2']} | {r['error_range']:.3f} | "
                f"{r['max_error']:.3f} | {r['min_error']:.3f} |"
            )
        lines.append("")

    # ──────────────────────────────────────────────────────────────────
    #  §10. SYNTHESIS & RECOMMENDATIONS
    # ──────────────────────────────────────────────────────────────────
    lines.append(_section("10. Synthesis & Recommendations"))

    lines.append(_build_synthesis(kwargs))

    # ── Write ─────────────────────────────────────────────────────────
    report_text = "\n".join(lines)
    report_path = out_dir / "rapport_error_analysis.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Also write plain text version for backward compatibility
    txt_path = out_dir / "rapport_error_analysis.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n  ✓ Report: {report_path}")
    print(f"  ✓ Report (txt): {txt_path}")

    return report_text


def _build_synthesis(kwargs):
    """Build the synthesis section based on available results."""
    lines = []

    lines.append("")
    lines.append("### Key Findings")
    lines.append("")

    findings = []

    # Finding 1: Dominant error source
    df = kwargs.get("df")
    label_errors_df = kwargs.get("label_errors_df")
    if label_errors_df is not None and not label_errors_df.empty:
        worst = label_errors_df.sort_values("FN_rate", ascending=False).iloc[0]
        findings.append(
            f"1. **Dominant error label**: `{worst['label']}` has the "
            f"highest FN rate ({worst['FN_rate']:.3f}), indicating the "
            f"model frequently misses this emotion."
        )

    # Finding 2: Violations
    violations_df = kwargs.get("violations_df")
    if violations_df is not None:
        n = len(violations_df)
        emo_no_mode = violations_df.get("emotion_no_mode", pd.Series()).sum()
        if emo_no_mode > 0:
            findings.append(
                f"2. **Structural weakness**: {emo_no_mode}/{n} samples "
                f"({100*emo_no_mode/n:.1f}%) have an emotion predicted "
                f"without any expression mode — the most prevalent "
                f"annotation scheme violation."
            )

    # Finding 3: Density effect
    density_results = kwargs.get("density_results")
    if density_results:
        sp = density_results.get("spearman_global", {})
        rho = sp.get("rho", np.nan)
        if not np.isnan(rho):
            direction = "increases" if rho > 0 else "decreases"
            findings.append(
                f"3. **Density-error relationship**: Error {direction} "
                f"with label density (ρ={rho:.3f}), "
                f"{'confirming' if rho > 0 else 'refuting'} the "
                f"hypothesis that denser vectors degrade performance."
            )

    for f in findings:
        lines.append(f)
        lines.append("")

    # Recommendations
    lines.append("### Recommendations")
    lines.append("")
    lines.append("1. **Post-processing cascade**: Implement logical "
                  "rules after thresholding to enforce annotation "
                  "scheme consistency (∀ emotion → Emo=1; "
                  "∀ emotion → at least one mode).")
    lines.append("")
    lines.append("2. **Fixed threshold monitoring**: Keep the decision "
                  "threshold at 0.5 and monitor calibration drift through "
                  "the probability and Brier analyses.")
    lines.append("")
    lines.append("3. **Context disabled for OOD**: Based on prior sanity "
                  "checks, context should be disabled for OOD data "
                  "(−9pp coherence degradation).")
    lines.append("")
    lines.append("4. **Monitor density at inference time**: Inputs with "
                  "high label density (many concurrent emotions) should "
                  "be flagged for potential degraded performance.")
    lines.append("")

    return "\n".join(lines)
