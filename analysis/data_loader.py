# -*- coding: utf-8 -*-
"""
Data Loader — Loading, Cleaning, Feature Engineering
"""

import numpy as np
import pandas as pd

from . import config


# ═══════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _is_dirty_annotation(val):
    """Détecte les artefacts d'annotation (ex: 'File: scenario_...Majority: NULL')."""
    if not isinstance(val, str):
        return False
    return val.startswith("File: ") or "Majority: NULL" in val


def _clean_qualitative_column(series, valid_set=None):
    """Nettoie une colonne qualitative : strip, normalise, filtre les artefacts."""
    cleaned = series.copy()
    for i, val in enumerate(cleaned):
        if pd.isna(val):
            cleaned.iloc[i] = np.nan
            continue
        s = str(val).strip()
        if _is_dirty_annotation(s):
            cleaned.iloc[i] = np.nan
            continue
        if valid_set is not None:
            # Gérer les valeurs composées (ex: "victim/victim_support")
            parts = [p.strip() for p in s.split("/")]
            if all(p in valid_set for p in parts):
                cleaned.iloc[i] = s
            else:
                cleaned.iloc[i] = np.nan
        else:
            cleaned.iloc[i] = s
    return cleaned.astype("category")


def _clean_target_column(series, valid_roles):
    """Nettoie la colonne TARGET qui peut contenir des valeurs composées."""
    cleaned = series.copy()
    for i, val in enumerate(cleaned):
        if pd.isna(val):
            cleaned.iloc[i] = np.nan
            continue
        s = str(val).strip()
        if _is_dirty_annotation(s):
            cleaned.iloc[i] = np.nan
            continue
        # Extraire le premier rôle cible pour simplifier
        parts = [p.strip() for p in s.split("/")]
        valid_parts = [p for p in parts if p in valid_roles]
        if valid_parts:
            cleaned.iloc[i] = valid_parts[0]  # Premier rôle = cible principale
        else:
            cleaned.iloc[i] = np.nan
    return cleaned.astype("category")


def _normalize_gold_column_aliases(df):
    """Map known ASCII gold column names to canonical accented labels."""
    rename_map = {}
    for alias, canonical in config.GOLD_COLUMN_ALIASES.items():
        if alias not in df.columns:
            continue
        if canonical in df.columns:
            df[canonical] = df[canonical].combine_first(df[alias])
        else:
            rename_map[alias] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def load_and_clean_data(xlsx_path=None):
    xlsx_path = config.resolve_xlsx_path(xlsx_path)

    df_all = pd.read_excel(xlsx_path)
    df_all["_original_idx"] = range(len(df_all))
    print(f"  ✓ {xlsx_path}: {len(df_all)} lignes, {len(df_all.columns)} colonnes")

    df_all = _normalize_gold_column_aliases(df_all)
    n_total = len(df_all)
    print(f"\n  Total : {n_total} lignes")

    # ── Extraction of nature_linguistique ─────────────────────────────
    nl_cols = [f"nature_linguistique_span_{i}" for i in range(1, 5)]
    nl_cols_present = [c for c in nl_cols if c in df_all.columns]
    if nl_cols_present:
        df_all["nature_linguistique"] = df_all[nl_cols_present].bfill(axis=1).iloc[:, 0]

    # ── Nettoyage des features qualitatives ───────────────────────────
    for col in config.QUALITATIVE_FEATURES:
        if col not in df_all.columns:
            continue
        if col == "TARGET":
            df_all[col] = _clean_target_column(
                df_all[col], config.VALID_VALUES.get("TARGET", set())
            )
        else:
            df_all[col] = _clean_qualitative_column(
                df_all[col], config.VALID_VALUES.get(col)
            )

    # ── Nettoyage des features binaires ───────────────────────────────
    for col in config.BINARY_FEATURES:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce").fillna(0).astype(int)

    # ── Nettoyage des gold labels ─────────────────────────────────────
    for gold_col in config.FULL_GOLD_TO_EMOTYC:
        if gold_col in df_all.columns:
            df_all[gold_col] = pd.to_numeric(df_all[gold_col], errors="coerce").fillna(0)
            df_all[gold_col] = (df_all[gold_col] >= 0.5).astype(int)

    # ── Colonne TEXT ──────────────────────────────────────────────────
    text_col = None
    for candidate in ("TEXT", "text", "sentence"):
        if candidate in df_all.columns:
            text_col = candidate
            break
    if text_col is None:
        raise ValueError("Colonne texte introuvable (TEXT/text/sentence)")
    if text_col != "TEXT":
        df_all = df_all.rename(columns={text_col: "TEXT"})
    df_all["TEXT"] = df_all["TEXT"].fillna("").astype(str)

    n_clean = (
        df_all[config.QUALITATIVE_FEATURES[0]].notna().sum()
        if config.QUALITATIVE_FEATURES[0] in df_all.columns
        else n_total
    )
    print(f"  Après nettoyage : {n_clean}/{n_total} lignes avec "
          f"{config.QUALITATIVE_FEATURES[0]} valide")

    return df_all


def add_text_features(df):
    """
    Ajoute des features textuelles dérivées.

    Features added: text_length, word_count, pct_uppercase,
                    has_exclamation, has_question,
                    mean_span_avg_tok_len, n_frag_words_in_spans,
                    mean_text_avg_tok_len, n_frag_words_in_text,
                    n_text_elongated_words,
                    ratio_text_elongated_words
    """
    import re

    df["text_length"] = df["TEXT"].str.len()
    df["word_count"] = df["TEXT"].str.split().str.len()
    df["pct_uppercase"] = df["TEXT"].apply(
        lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1)
    )
    df["has_exclamation"] = df["TEXT"].str.contains("!").astype(int)
    df["has_question"] = df["TEXT"].str.contains(r"\?").astype(int)

    word_pattern = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", flags=re.UNICODE)
    repeated_char_pattern = re.compile(r"([^\W\d_])\1{2,}", flags=re.UNICODE)

    common_short_words = {
        "a", "à", "c", "d", "j", "l", "m", "n", "s", "t", "y",
        "ai", "as", "au", "ça", "ca", "ce", "de", "du", "en", "es",
        "et", "eu", "il", "je", "la", "le", "ma", "me", "ne", "ni",
        "on", "ou", "où", "sa", "se", "si", "ta", "te", "tu", "un",
        "va", "vu",
    }
    # ── Tokenization quality of spans ──────────────────────────────
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("camembert-base", use_fast=True)

        def extract_words(text):
            if pd.isna(text):
                return []
            return word_pattern.findall(str(text).lower())

        def normalize_word(word):
            return word.replace("’", "'").strip("'")

        def should_score_fragmentation(word):
            normalized = normalize_word(word)
            if "'" in normalized:
                return False
            letters_only = re.sub(r"[\W\d_]", "", normalized, flags=re.UNICODE)
            if not letters_only:
                return False
            if len(letters_only) >= 3:
                return True
            return letters_only not in common_short_words

        def is_elongated_word(word):
            return bool(repeated_char_pattern.search(normalize_word(word)))

        def compute_fragmentation(text):
            words = set(extract_words(text))

            n_frag = 0
            total_avg = []

            for word in words:
                if not should_score_fragmentation(word):
                    continue
                n_chars = len(re.sub(r"[\W\d_]", "", normalize_word(word), flags=re.UNICODE))
                tok_ids = tokenizer.encode(word, add_special_tokens=False)
                if not tok_ids:
                    continue
                avg_tok_len = n_chars / len(tok_ids)
                total_avg.append(avg_tok_len)
                if avg_tok_len <= 1.5:
                    n_frag += 1

            mean_avg_tok_len = sum(total_avg) / len(total_avg) if total_avg else np.nan
            return pd.Series([mean_avg_tok_len, n_frag])

        def compute_text_irregularity(text):
            words = extract_words(text)
            if not words:
                return pd.Series([0, 0.0])

            n_elongated = 0

            for word in words:
                normalized = normalize_word(word)
                letters_only = re.sub(r"[\W\d_]", "", normalized, flags=re.UNICODE)
                if not letters_only:
                    continue

                elongated = is_elongated_word(letters_only)

                n_elongated += int(elongated)

            n_words = len(words)
            return pd.Series([
                n_elongated,
                n_elongated / n_words,
            ])

        def compute_span_fragmentation(row):
            span_texts = [
                str(row.get(f"span{n}_text", ""))
                for n in range(1, 5)
                if pd.notna(row.get(f"span{n}_text"))
            ]
            if not span_texts:
                return pd.Series([np.nan, 0])

            combined_spans = " ".join(span_texts).lower()
            return compute_fragmentation(combined_spans)

        res = df.apply(compute_span_fragmentation, axis=1)
        df["mean_span_avg_tok_len"] = res[0]
        df["n_frag_words_in_spans"] = res[1].fillna(0).astype(int)

        text_res = df["TEXT"].apply(compute_fragmentation)
        df["mean_text_avg_tok_len"] = text_res[0]
        df["n_frag_words_in_text"] = text_res[1].fillna(0).astype(int)

        irregularity_res = df["TEXT"].apply(compute_text_irregularity)
        df["n_text_elongated_words"] = irregularity_res[0].astype(int)
        df["ratio_text_elongated_words"] = irregularity_res[1]
    except Exception as e:
        print(f"Erreur calcul tokénisation: {e}")
        df["mean_span_avg_tok_len"] = np.nan
        df["n_frag_words_in_spans"] = 0
        df["mean_text_avg_tok_len"] = np.nan
        df["n_frag_words_in_text"] = 0
        df["n_text_elongated_words"] = 0
        df["ratio_text_elongated_words"] = 0.0

    return df


def add_density_features(df):
    emo_cols = [e for e in config.EMOTION_12 if e in df.columns]
    mode_cols = [m for m in config.MODES_4 if m in df.columns]
    all_cols = [l for l in config.ALL_19 if l in df.columns]

    if emo_cols:
        df["emotion_density_12"] = df[emo_cols].sum(axis=1)
    if mode_cols:
        df["mode_density_4"] = df[mode_cols].sum(axis=1)
    if all_cols:
        df["label_density_19"] = df[all_cols].sum(axis=1)

    # Prediction densities (if pred columns exist)
    pred_emo_cols = [f"pred_{e}" for e in config.EMOTION_12 if f"pred_{e}" in df.columns]
    pred_mode_cols = [f"pred_{m}" for m in config.MODES_4 if f"pred_{m}" in df.columns]
    if pred_emo_cols:
        df["pred_emotion_density"] = df[pred_emo_cols].sum(axis=1)
    if pred_mode_cols:
        df["pred_mode_density"] = df[pred_mode_cols].sum(axis=1)

    return df


def build_analysis_features(df):
    """
    Construit la matrice de features explicatives pour le modèle de diagnostic.

    Returns
    -------
    X_df : pd.DataFrame
        Numeric feature matrix (no NaN).
    feature_names : list[str]
        Ordered feature names.
    """
    parts = []
    feature_names = []

    # A) Features binaires
    for col in config.BINARY_FEATURES:
        if col in df.columns:
            parts.append(df[col].fillna(0).astype(int).values.reshape(-1, 1))
            feature_names.append(col)

    # B) Features textuelles dérivées
    for col in config.TEXT_FEATURES:
        if col in df.columns:
            parts.append(df[col].fillna(0).values.reshape(-1, 1))
            feature_names.append(col)

    # C) Features qualitatives (one-hot, with MISSING category)
    for col in config.QUALITATIVE_FEATURES:
        if col not in df.columns:
            continue
        series = df[col].astype(str).replace("nan", "MISSING").fillna("MISSING")
        dummies = pd.get_dummies(series, prefix=col)
        parts.append(dummies.values)
        feature_names.extend(dummies.columns.tolist())

    if not parts:
        X_df = pd.DataFrame(index=df.index)
        print(f"  Matrice de features : {X_df.shape[0]} lignes × 0 features")
        return X_df, feature_names

    X = np.hstack(parts)
    X_df = pd.DataFrame(X, columns=feature_names, index=df.index)
    print(f"  Matrice de features : {X_df.shape[0]} lignes × {X_df.shape[1]} features")
    return X_df, feature_names
