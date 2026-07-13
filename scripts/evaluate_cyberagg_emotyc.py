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
THRESHOLD = 0.5
BATCH_SIZE = 32


@dataclass(frozen=True)
class RunSpec:
    template_label: str
    model_label: str
    model_alias: str
    template: str
    use_context: bool


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
    rows = [run_evaluation(data_path, spec) for spec in RUNS]
    print(format_markdown_table(rows))


def run_evaluation(data_path: Path, spec: RunSpec) -> dict[str, object]:
    bundle = resolve_model_bundle(spec.model_alias)
    predictor = Predictor.from_bundle(bundle)
    dataset = load_xlsx(data_path, model_labels=predictor.labels, require_gold=True)
    if len(dataset.texts) != EXPECTED_SAMPLES:
        raise ValueError(f"{DATASET} has {len(dataset.texts)} samples; expected {EXPECTED_SAMPLES}")
    if dataset.gold is None:
        raise ValueError(f"{DATASET} does not contain usable gold labels")

    texts = apply_template(dataset.texts, spec.template, use_context=spec.use_context)
    result = predictor.predict(texts, batch_size=BATCH_SIZE, threshold=THRESHOLD)
    indices = [predictor.labels.index(label) for label in dataset.labels_evaluated]
    predictions = result.predictions[:, indices]
    global_metrics, _ = compute_metrics(dataset.gold, predictions, dataset.labels_evaluated)
    return {
        "template": spec.template_label,
        "model": spec.model_label,
        "micro_f1": global_metrics["micro_f1"],
        "macro_f1": global_metrics["macro_f1"],
        "exact_match": global_metrics["exact_match"],
    }


def format_markdown_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Template | Modèle | micro-F1 | macro-F1 | exact match |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {template} | {model} | {micro_f1:.4f} | {macro_f1:.4f} | {exact_match:.4f} |".format(
                **row
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
