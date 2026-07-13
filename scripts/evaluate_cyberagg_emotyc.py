#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from emotyc.artifacts import resolve_model_bundle
from emotyc.data_sources import resolve_dataset
from emotyc.datasets import load_xlsx
from emotyc.formatting import apply_template
from emotyc.metrics import compute_metrics
from emotyc.predictors import Predictor


DATASET = "CyberAgg"
EXPECTED_SAMPLES = 783
EXPECTED_LABELS = 19
THRESHOLD = 0.5
BATCH_SIZE = 128
ADD_SPECIAL_TOKENS = False


@dataclass(frozen=True)
class RunSpec:
    template_label: str
    model_label: str
    model_alias: str
    template: str
    use_context: bool


@dataclass(frozen=True)
class EvaluationResult:
    spec: RunSpec
    global_metrics: dict[str, object]
    per_label: list[dict[str, object]]


RUNS = (
    RunSpec("raw", "EMOTYC_1", "emotyc_1", "raw", False),
    RunSpec("raw", "EMOTYC_2", "emotyc_2", "raw", False),
    RunSpec("bca", "EMOTYC_1", "emotyc_1", "bca", False),
    RunSpec("bca", "EMOTYC_2", "emotyc_2", "bca", False),
    RunSpec("bca-context", "EMOTYC_1", "emotyc_1", "bca", True),
    RunSpec("bca-context", "EMOTYC_2", "emotyc_2", "bca", True),
)


def main() -> None:
    data_path = resolve_dataset(DATASET)
    results = [run_evaluation(data_path, spec) for spec in RUNS]

    print(f"add_special_tokens = {ADD_SPECIAL_TOKENS}")
    print("Métriques globales")
    print(format_global_table(results))
    print("")
    print("Métriques détaillées par label")
    print(format_per_label_table(results))


def run_evaluation(data_path: Path, spec: RunSpec) -> EvaluationResult:
    bundle = resolve_model_bundle(spec.model_alias)
    predictor = Predictor.from_bundle(
        bundle,
        add_special_tokens=ADD_SPECIAL_TOKENS,
    )
    dataset = load_xlsx(data_path, model_labels=predictor.labels, require_gold=True)
    if len(dataset.texts) != EXPECTED_SAMPLES:
        raise ValueError(
            f"{DATASET} contient {len(dataset.texts)} échantillons ; {EXPECTED_SAMPLES} attendus."
        )
    if dataset.gold is None:
        raise ValueError(f"{DATASET} ne contient aucun label gold exploitable.")
    if len(dataset.labels_evaluated) != EXPECTED_LABELS:
        raise ValueError(
            f"{DATASET} contient {len(dataset.labels_evaluated)} labels évaluables pour "
            f"{spec.model_label} ; {EXPECTED_LABELS} attendus."
        )

    texts = apply_template(dataset.texts, spec.template, use_context=spec.use_context)
    result = predictor.predict(texts, batch_size=BATCH_SIZE, threshold=THRESHOLD)
    indices = [predictor.labels.index(label) for label in dataset.labels_evaluated]
    predictions = result.predictions[:, indices]
    global_metrics, per_label = compute_metrics(
        dataset.gold, predictions, dataset.labels_evaluated
    )
    return EvaluationResult(spec, global_metrics, per_label)


def format_global_table(results: list[EvaluationResult]) -> str:
    lines = [
        "| Template | Modèle | micro-F1 | macro-F1 | exact match |",
        "|---|---|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| {template} | {model} | {micro_f1:.4f} | {macro_f1:.4f} | "
            "{exact_match:.4f} |".format(
                template=result.spec.template_label,
                model=result.spec.model_label,
                **result.global_metrics,
            )
        )
    return "\n".join(lines)


def format_per_label_table(results: list[EvaluationResult]) -> str:
    lines = [
        "| Template | Modèle | Label | Précision | Rappel | F1 | Support |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for result in results:
        for row in result.per_label:
            lines.append(
                "| {template} | {model} | {label} | {precision:.4f} | {recall:.4f} | "
                "{f1:.4f} | {support} |".format(
                    template=result.spec.template_label,
                    model=result.spec.model_label,
                    **row,
                )
            )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
