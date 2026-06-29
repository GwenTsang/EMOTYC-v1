from __future__ import annotations

from pathlib import Path

from emotyc.io import read_json


def load_metrics(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"metrics.json not found: {path}")
    data = read_json(path)
    required = ("global_metrics", "per_label", "labels_evaluated")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Invalid metrics.json {path}: missing {', '.join(missing)}")
    return data


def compare_metrics(path_a: str | Path, path_b: str | Path) -> dict:
    a = load_metrics(path_a)
    b = load_metrics(path_b)
    global_common = sorted(set(a["global_metrics"]) & set(b["global_metrics"]))
    global_deltas = {
        key: _delta(a["global_metrics"][key], b["global_metrics"][key])
        for key in global_common
        if _is_number(a["global_metrics"][key]) and _is_number(b["global_metrics"][key])
    }
    a_labels = {row["label"]: row for row in a["per_label"]}
    b_labels = {row["label"]: row for row in b["per_label"]}
    labels_a = set(a["labels_evaluated"])
    labels_b = set(b["labels_evaluated"])
    common_labels = sorted(labels_a & labels_b)
    per_label = {}
    for label in common_labels:
        if label not in a_labels or label not in b_labels:
            continue
        per_label[label] = {
            metric: _delta(a_labels[label][metric], b_labels[label][metric])
            for metric in ("precision", "recall", "f1", "support")
            if metric in a_labels[label] and metric in b_labels[label]
            and _is_number(a_labels[label][metric]) and _is_number(b_labels[label][metric])
        }
    return {
        "global": global_deltas,
        "per_label": per_label,
        "labels_common": common_labels,
        "labels_only_a": sorted(labels_a - labels_b),
        "labels_only_b": sorted(labels_b - labels_a),
    }


def _delta(a_value: int | float, b_value: int | float) -> dict:
    return {"a": a_value, "b": b_value, "delta": round(b_value - a_value, 4)}


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float))
