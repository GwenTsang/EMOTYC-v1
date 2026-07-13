#!/usr/bin/env python
"""Run the six CyberAgg evaluations and relate dataset features to errors.

This is intentionally a self-contained experiment script, not a reusable
pipeline.

SHAP explains a Random-Forest surrogate trained to predict each run's row-wise
Hamming error.  It therefore describes associations with EMOTYC errors; it is
not an explanation of the internal ONNX model.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from emotyc.artifacts import resolve_model_bundle
from emotyc.data_sources import resolve_dataset
from emotyc.datasets import load_xlsx
from emotyc.formatting import apply_template
from emotyc.metrics import compute_metrics
from emotyc.predictors import Predictor
from evaluate_cyberagg_emotyc import (
    BATCH_SIZE,
    DATASET,
    EXPECTED_LABELS,
    EXPECTED_SAMPLES,
    RUNS,
)


THRESHOLD = 0.5
ADD_SPECIAL_TOKENS = False

BINARY_FEATURES = [
    "elongation",
    "ironie",
    "insulte",
    "mépris / haine",
    "argot",
    "abréviation",
    "interjection",
]
QUALITATIVE_FEATURES = [
    "ROLE",
    "HATE",
    "TARGET",
    "VERBAL_ABUSE",
    "INTENTION",
    "CONTEXT",
    "SENTIMENT",
    "nature_linguistique",
]
TEXT_FEATURES = [
    "text_length",
    "mean_span_avg_tok_len",
    "n_frag_words_in_spans",
    "mean_text_avg_tok_len",
    "n_frag_words_in_text",
    "n_text_elongated_words",
    "ratio_text_elongated_words",
]
VALID_VALUES = {
    "ROLE": {"bully", "victim", "bully_support", "victim_support", "conciliator"},
    "HATE": {"OAG", "CAG", "NAG"},
    "SENTIMENT": {"POS", "NEG", "NEU"},
    "TARGET": {"bully", "victim", "bully_support", "victim_support", "conciliator"},
    "VERBAL_ABUSE": {"BLM", "NCG", "THR", "DNG", "OTH"},
    "INTENTION": {"ATK", "DFN", "CNS", "AIN", "GSL", "EMP", "CR", "OTH"},
    "CONTEXT": {"ATK", "DFN", "CNS", "AIN", "GSL", "EMP", "CR", "OTH"},
}

WORD_PATTERN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", flags=re.UNICODE)
REPEATED_CHAR_PATTERN = re.compile(r"([^\W\d_])\1{2,}", flags=re.UNICODE)
COMMON_SHORT_WORDS = {
    "a", "à", "c", "d", "j", "l", "m", "n", "s", "t", "y",
    "ai", "as", "au", "ça", "ca", "ce", "de", "du", "en", "es",
    "et", "eu", "il", "je", "la", "le", "ma", "me", "ne", "ni",
    "on", "ou", "où", "sa", "se", "si", "ta", "te", "tu", "un",
    "va", "vu",
}


@dataclass
class RunOutput:
    run_id: str
    model: str
    template: str
    use_context: bool
    predictions: np.ndarray
    probabilities: np.ndarray
    global_metrics: dict[str, object]
    per_label: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exécuter les 6 runs CyberAgg et expliquer leurs erreurs."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/cyberagg_error_analysis"),
        help="Dossier des prédictions et résultats d'analyse.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def _clean_qualitative(value: object, feature: str) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.startswith("File: ") or "Majority: NULL" in text:
        return np.nan
    valid = VALID_VALUES.get(feature)
    if valid is None:
        return text
    parts = [part.strip() for part in text.split("/")]
    if not parts or not all(part in valid for part in parts):
        return np.nan
    # The historical analysis keeps only the primary TARGET.
    return parts[0] if feature == "TARGET" else "/".join(parts)


def _first_existing_columns(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    present = [column for column in candidates if column in frame.columns]
    if not present:
        return pd.Series(np.nan, index=frame.index)
    return frame[present].bfill(axis=1).iloc[:, 0]


def _token_count(tokenizer: object, word: str) -> int:
    encoded = tokenizer.encode(word, add_special_tokens=False)
    ids = encoded.ids if hasattr(encoded, "ids") else encoded
    return len(ids)


def _fragmentation(text: object, tokenizer: object) -> tuple[float, int]:
    """Historical CamemBERT fragmentation metric from analysis/data_loader.py."""
    words = set(WORD_PATTERN.findall(str(text).lower())) if not pd.isna(text) else set()
    average_token_lengths: list[float] = []
    n_fragmented = 0
    for word in words:
        normalized = word.replace("’", "'").strip("'")
        if "'" in normalized:
            continue
        letters = re.sub(r"[\W\d_]", "", normalized, flags=re.UNICODE)
        if not letters or (len(letters) < 3 and letters in COMMON_SHORT_WORDS):
            continue
        n_tokens = _token_count(tokenizer, word)
        if n_tokens == 0:
            continue
        average_length = len(letters) / n_tokens
        average_token_lengths.append(average_length)
        n_fragmented += int(average_length <= 1.5)
    mean_length = float(np.mean(average_token_lengths)) if average_token_lengths else np.nan
    return mean_length, n_fragmented


def build_features(frame: pd.DataFrame, tokenizer: object) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    features["sample_id"] = np.arange(len(frame))

    # idx restarts at each of the four CyberAgg scenarios.
    idx_values = pd.to_numeric(frame["idx"], errors="coerce")
    features["scenario"] = (idx_values.diff().fillna(1) <= 0).cumsum().astype(int)

    for feature in BINARY_FEATURES:
        features[feature] = pd.to_numeric(frame[feature], errors="coerce").fillna(0).astype(int)

    nature = _first_existing_columns(
        frame,
        [
            *(f"Sit_Emo_unit_{index}_nature_linguistique" for index in range(1, 5)),
            *(f"nature_linguistique_span_{index}" for index in range(1, 5)),
        ],
    )
    for feature in QUALITATIVE_FEATURES:
        source = nature if feature == "nature_linguistique" else frame[feature]
        features[feature] = source.map(lambda value, name=feature: _clean_qualitative(value, name))

    text = frame["TEXT"].fillna("").astype(str)
    features["text_length"] = text.str.len()

    span_columns = [
        column
        for index in range(1, 5)
        for column in (f"Sit_Emo_unit_{index}_segment_text", f"span{index}_text")
        if column in frame.columns
    ]

    def span_text(row: pd.Series) -> str:
        return " ".join(str(row[column]) for column in span_columns if pd.notna(row[column]))

    span_fragmentation = frame.apply(
        lambda row: _fragmentation(span_text(row), tokenizer), axis=1, result_type="expand"
    )
    text_fragmentation = text.map(lambda value: _fragmentation(value, tokenizer))
    features["mean_span_avg_tok_len"] = span_fragmentation[0].astype(float)
    features["n_frag_words_in_spans"] = span_fragmentation[1].fillna(0).astype(int)
    features["mean_text_avg_tok_len"] = text_fragmentation.map(lambda value: value[0])
    features["n_frag_words_in_text"] = text_fragmentation.map(lambda value: value[1]).astype(int)

    def elongated(text_value: str) -> tuple[int, float]:
        words = WORD_PATTERN.findall(text_value.lower())
        count = sum(
            bool(REPEATED_CHAR_PATTERN.search(word.replace("’", "'").strip("'")))
            for word in words
        )
        return count, count / len(words) if words else 0.0

    irregularity = text.map(elongated)
    features["n_text_elongated_words"] = irregularity.map(lambda value: value[0])
    features["ratio_text_elongated_words"] = irregularity.map(lambda value: value[1])
    return features


def build_model_matrix(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    columns: list[pd.Series | pd.DataFrame] = []
    groups: dict[str, list[str]] = {}
    for feature in BINARY_FEATURES + TEXT_FEATURES:
        values = pd.to_numeric(features[feature], errors="coerce")
        median = values.median() if values.notna().any() else 0.0
        series = values.fillna(median).astype(float).rename(feature)
        columns.append(series)
        groups[feature] = [feature]
    for feature in QUALITATIVE_FEATURES:
        values = features[feature].astype("string").fillna("MISSING")
        dummies = pd.get_dummies(values, prefix=feature, dtype=float)
        columns.append(dummies)
        groups[feature] = dummies.columns.tolist()
    matrix = pd.concat(columns, axis=1)
    varying = matrix.nunique(dropna=False) > 1
    matrix = matrix.loc[:, varying]
    groups = {
        feature: [column for column in names if column in matrix]
        for feature, names in groups.items()
    }
    return matrix, groups


def bh_adjust(p_values: pd.Series) -> np.ndarray:
    values = p_values.fillna(1.0).to_numpy(float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted


def univariate_associations(
    features: pd.DataFrame, errors: np.ndarray, run_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    levels: list[dict[str, object]] = []
    for feature in BINARY_FEATURES + TEXT_FEATURES:
        values = pd.to_numeric(features[feature], errors="coerce")
        valid = values.notna()
        if valid.sum() < 3 or values[valid].nunique() < 2:
            rho, p_value = 0.0, 1.0
        else:
            rho, p_value = stats.spearmanr(values[valid], errors[valid], nan_policy="omit")
        rows.append(
            {
                "run_id": run_id,
                "feature": feature,
                "kind": "numeric",
                "effect": float(rho),
                "p_value": float(p_value),
            }
        )
        if feature in BINARY_FEATURES or values.nunique() <= 6:
            for level, index in values.groupby(values, dropna=False).groups.items():
                levels.append(
                    {
                        "run_id": run_id,
                        "feature": feature,
                        "level": level,
                        "n": len(index),
                        "mean_hamming_error": float(np.mean(errors[np.asarray(index)])),
                    }
                )

    for feature in QUALITATIVE_FEATURES:
        values = features[feature].astype("string").fillna("MISSING")
        grouped = [errors[index] for index in values.groupby(values).groups.values() if len(index) >= 3]
        if len(grouped) < 2:
            statistic, p_value, epsilon_squared = 0.0, 1.0, 0.0
        else:
            statistic, p_value = stats.kruskal(*grouped)
            n = sum(len(group) for group in grouped)
            epsilon_squared = max((statistic - len(grouped) + 1) / (n - len(grouped)), 0.0)
        rows.append(
            {
                "run_id": run_id,
                "feature": feature,
                "kind": "categorical",
                "effect": float(epsilon_squared),
                "p_value": float(p_value),
            }
        )
        for level, index in values.groupby(values).groups.items():
            levels.append(
                {
                    "run_id": run_id,
                    "feature": feature,
                    "level": level,
                    "n": len(index),
                    "mean_hamming_error": float(np.mean(errors[np.asarray(index)])),
                }
            )
    associations = pd.DataFrame(rows)
    associations["q_value_bh"] = bh_adjust(associations["p_value"])
    return associations, pd.DataFrame(levels)


def rf_shap(
    matrix: pd.DataFrame,
    groups: dict[str, list[str]],
    errors: np.ndarray,
    scenarios: np.ndarray,
    run_id: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    parameters = {
        "n_estimators": 400,
        "max_depth": 7,
        "min_samples_leaf": 10,
        "random_state": 42,
        "n_jobs": -1,
    }
    splitter = GroupKFold(n_splits=len(np.unique(scenarios)))
    cv_predictions = np.empty(len(errors), dtype=float)
    for train, test in splitter.split(matrix, errors, groups=scenarios):
        fold_model = RandomForestRegressor(**parameters)
        fold_model.fit(matrix.iloc[train], errors[train])
        cv_predictions[test] = fold_model.predict(matrix.iloc[test])

    model = RandomForestRegressor(**parameters, oob_score=True)
    model.fit(matrix, errors)
    diagnostics = {
        "run_id": run_id,
        "cv_scheme": "leave-one-scenario-out (4 folds)",
        "cv_r2": float(r2_score(errors, cv_predictions)),
        "cv_mae": float(mean_absolute_error(errors, cv_predictions)),
        "oob_r2": float(model.oob_score_),
    }

    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP manque : exécutez `python -m pip install shap`.") from exc
    shap_values = np.asarray(shap.TreeExplainer(model).shap_values(matrix))
    importance_rows = []
    for feature, encoded_columns in groups.items():
        if not encoded_columns:
            continue
        indices = [matrix.columns.get_loc(column) for column in encoded_columns]
        grouped_values = shap_values[:, indices].sum(axis=1)
        importance_rows.append(
            {
                "run_id": run_id,
                "feature": feature,
                "mean_abs_shap": float(np.mean(np.abs(grouped_values))),
                "mean_signed_shap": float(np.mean(grouped_values)),
            }
        )
    importance = pd.DataFrame(importance_rows).sort_values("mean_abs_shap", ascending=False)
    importance["rank"] = np.arange(1, len(importance) + 1)
    return importance, diagnostics


def run_id(model_alias: str, template_label: str) -> str:
    return f"{model_alias}_{template_label.replace('-', '_')}"


def save_predictions(
    path: Path,
    frame: pd.DataFrame,
    labels: list[str],
    gold: np.ndarray,
    output: RunOutput,
) -> None:
    saved = pd.DataFrame(
        {
            "sample_id": np.arange(len(frame)),
            "idx": frame["idx"],
            "ID": frame["ID"],
            "TEXT": frame["TEXT"].fillna(""),
        }
    )
    for index, label in enumerate(labels):
        saved[f"gold_{label}"] = gold[:, index]
        saved[f"pred_{label}"] = output.predictions[:, index]
        saved[f"proba_{label}"] = output.probabilities[:, index]
    saved.to_csv(path, index=False)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Aucun résultat._"
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in frame.itertuples(index=False, name=None):
        formatted = []
        for value in row:
            formatted.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def write_summary(
    path: Path,
    global_metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    importance: pd.DataFrame,
    associations: pd.DataFrame,
    add_special_tokens: bool,
) -> None:
    global_view = global_metrics[
        ["run_id", "micro_f1", "macro_f1", "exact_match", "mean_hamming_error"]
    ].copy()
    consensus = (
        importance.assign(top5=importance["rank"] <= 5)
        .groupby("feature", as_index=False)
        .agg(top5_runs=("top5", "sum"), mean_abs_shap=("mean_abs_shap", "mean"))
        .sort_values(["top5_runs", "mean_abs_shap"], ascending=False)
        .head(12)
    )
    significant = associations[associations["q_value_bh"] < 0.05].copy()
    significant = (
        significant.sort_values(["run_id", "q_value_bh"])
        .groupby("run_id", sort=False)
        .head(5)
    )
    significant = significant[["run_id", "feature", "kind", "effect", "q_value_bh"]]
    lines = [
        "# Analyse des erreurs EMOTYC sur CyberAgg",
        "",
        "- Six runs : deux modèles × `raw`, `bca`, `bca-context`.",
        f"- `add_special_tokens={add_special_tokens}` ; erreur = taux de Hamming sur 19 labels.",
        "- SHAP explique un Random Forest substitut de l'erreur, pas le modèle EMOTYC directement.",
        "- Les q-values sont corrigées par Benjamini–Hochberg séparément pour chaque run.",
        "",
        "## Performances et erreur moyenne",
        "",
        markdown_table(global_view),
        "",
        "## Fidélité hors scénario des substituts",
        "",
        markdown_table(diagnostics[["run_id", "cv_r2", "cv_mae", "oob_r2"]]),
        "",
        "Un R² CV nul ou négatif interdit de lire le classement SHAP comme une explication robuste hors scénario.",
        "",
        "## Features les plus récurrentes dans SHAP",
        "",
        markdown_table(consensus),
        "",
        "## Associations univariées qui restent significatives après correction",
        "",
        markdown_table(significant),
        "",
        "Pour les variables numériques, `effect` est le rho de Spearman signé. Pour les variables "
        "qualitatives, il s'agit de l'epsilon carré de Kruskal–Wallis, toujours positif.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size doit être strictement positif")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_path = resolve_dataset(DATASET)
    first_bundle = resolve_model_bundle("emotyc_1")
    tokenizer_predictor = Predictor.from_bundle(
        first_bundle, add_special_tokens=ADD_SPECIAL_TOKENS
    )
    dataset = load_xlsx(data_path, model_labels=tokenizer_predictor.labels, require_gold=True)
    labels = dataset.labels_evaluated
    if len(dataset.texts) != EXPECTED_SAMPLES or len(labels) != EXPECTED_LABELS:
        raise ValueError(
            f"CyberAgg inattendu: {len(dataset.texts)} lignes et {len(labels)} labels"
        )
    if dataset.gold is None:
        raise ValueError("Colonnes gold absentes")

    features = build_features(dataset.frame, tokenizer_predictor.encoder.tokenizer)
    features.to_csv(args.out_dir / "features.csv", index=False)
    matrix, feature_groups = build_model_matrix(features)

    outputs: list[RunOutput] = []
    for model_alias in ("emotyc_1", "emotyc_2"):
        predictor = (
            tokenizer_predictor
            if model_alias == "emotyc_1"
            else Predictor.from_bundle(
                resolve_model_bundle(model_alias),
                add_special_tokens=ADD_SPECIAL_TOKENS,
            )
        )
        for spec in (item for item in RUNS if item.model_alias == model_alias):
            identifier = run_id(model_alias, spec.template_label)
            print(f"[{identifier}] inférence…", flush=True)
            texts = apply_template(dataset.texts, spec.template, use_context=spec.use_context)
            prediction = predictor.predict(
                texts, batch_size=args.batch_size, threshold=THRESHOLD
            )
            indices = [predictor.labels.index(label) for label in labels]
            predicted = prediction.predictions[:, indices]
            probabilities = prediction.probabilities[:, indices]
            metrics, per_label = compute_metrics(dataset.gold, predicted, labels)
            output = RunOutput(
                run_id=identifier,
                model=model_alias,
                template=spec.template_label,
                use_context=spec.use_context,
                predictions=predicted,
                probabilities=probabilities,
                global_metrics=metrics,
                per_label=per_label,
            )
            outputs.append(output)
            save_predictions(
                args.out_dir / f"predictions_{identifier}.csv",
                dataset.frame,
                labels,
                dataset.gold,
                output,
            )

    metric_rows = []
    per_label_rows = []
    error_rows = []
    association_frames = []
    level_frames = []
    importance_frames = []
    diagnostic_rows = []
    scenarios = features["scenario"].to_numpy()
    for output in outputs:
        differences = output.predictions != dataset.gold
        hamming = differences.mean(axis=1)
        false_positive = ((output.predictions == 1) & (dataset.gold == 0)).sum(axis=1)
        false_negative = ((output.predictions == 0) & (dataset.gold == 1)).sum(axis=1)
        metric_rows.append(
            {
                "run_id": output.run_id,
                "model": output.model,
                "template": output.template,
                **output.global_metrics,
                "mean_hamming_error": float(hamming.mean()),
            }
        )
        per_label_rows.extend(
            {"run_id": output.run_id, **row} for row in output.per_label
        )
        error_rows.extend(
            {
                "sample_id": index,
                "run_id": output.run_id,
                "hamming_error": hamming[index],
                "n_errors": int(differences[index].sum()),
                "any_error": int(differences[index].any()),
                "n_false_positives": int(false_positive[index]),
                "n_false_negatives": int(false_negative[index]),
            }
            for index in range(len(dataset.texts))
        )
        associations, levels = univariate_associations(features, hamming, output.run_id)
        association_frames.append(associations)
        level_frames.append(levels)
        shap_importance, diagnostics = rf_shap(
            matrix, feature_groups, hamming, scenarios, output.run_id
        )
        importance_frames.append(shap_importance)
        diagnostic_rows.append(diagnostics)

    global_frame = pd.DataFrame(metric_rows)
    per_label_frame = pd.DataFrame(per_label_rows)
    errors_frame = pd.DataFrame(error_rows)
    associations_frame = pd.concat(association_frames, ignore_index=True)
    levels_frame = pd.concat(level_frames, ignore_index=True)
    global_frame.to_csv(args.out_dir / "global_metrics.csv", index=False)
    per_label_frame.to_csv(args.out_dir / "per_label_metrics.csv", index=False)
    errors_frame.to_csv(args.out_dir / "row_errors.csv", index=False)
    associations_frame.to_csv(args.out_dir / "univariate_associations.csv", index=False)
    levels_frame.to_csv(args.out_dir / "feature_level_errors.csv", index=False)

    manifest = {
        "dataset": DATASET,
        "dataset_path": str(data_path),
        "n_samples": len(dataset.texts),
        "labels": labels,
        "threshold": THRESHOLD,
        "batch_size": args.batch_size,
        "add_special_tokens": ADD_SPECIAL_TOKENS,
        "runs": [
            {
                "run_id": output.run_id,
                "model": output.model,
                "template": output.template,
                "use_context": output.use_context,
            }
            for output in outputs
        ],
        "feature_families": {
            "binary": BINARY_FEATURES,
            "qualitative": QUALITATIVE_FEATURES,
            "text": TEXT_FEATURES,
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if importance_frames:
        importance_frame = pd.concat(importance_frames, ignore_index=True)
        diagnostics_frame = pd.DataFrame(diagnostic_rows)
        importance_frame.to_csv(args.out_dir / "shap_importance.csv", index=False)
        diagnostics_frame.to_csv(args.out_dir / "surrogate_diagnostics.csv", index=False)
        write_summary(
            args.out_dir / "analysis_summary.md",
            global_frame,
            diagnostics_frame,
            importance_frame,
            associations_frame,
            ADD_SPECIAL_TOKENS,
        )

    print("\nMétriques globales")
    print(global_frame.to_string(index=False))
    print(f"\nRésultats écrits dans {args.out_dir}")

if __name__ == "__main__":
    main()
