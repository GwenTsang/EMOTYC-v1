from __future__ import annotations

import numpy as np


def precision_recall_f1_support(
    gold: np.ndarray,
    pred: np.ndarray,
    labels: list[str],
) -> list[dict]:
    """Compute per-label precision, recall, F1 and support."""
    gold, pred = _validate_inputs(gold, pred, labels)
    true_positive = ((gold == 1) & (pred == 1)).sum(axis=0)
    false_positive = ((gold == 0) & (pred == 1)).sum(axis=0)
    false_negative = ((gold == 1) & (pred == 0)).sum(axis=0)
    support = (gold == 1).sum(axis=0)

    rows = []
    for index, label in enumerate(labels):
        precision = _safe_div(true_positive[index], true_positive[index] + false_positive[index])
        recall = _safe_div(true_positive[index], true_positive[index] + false_negative[index])
        f1 = _f1(precision, recall)
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support[index]),
            }
        )
    return rows


def micro_f1(gold: np.ndarray, pred: np.ndarray) -> float:
    """Compute micro-F1 over all labels."""
    gold, pred = _validate_pair(gold, pred)
    true_positive = int(((gold == 1) & (pred == 1)).sum())
    false_positive = int(((gold == 0) & (pred == 1)).sum())
    false_negative = int(((gold == 1) & (pred == 0)).sum())
    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    return _f1(precision, recall)


def macro_f1(gold: np.ndarray, pred: np.ndarray, labels: list[str]) -> float:
    """Compute the unweighted mean of per-label F1 scores."""
    rows = precision_recall_f1_support(gold, pred, labels)
    if not rows:
        return 0.0
    return float(np.mean([row["f1"] for row in rows]))


def exact_match(gold: np.ndarray, pred: np.ndarray) -> float:
    """Compute exact row-level multi-label match."""
    gold, pred = _validate_pair(gold, pred)
    if gold.shape[0] == 0:
        return 0.0
    return float(np.all(gold == pred, axis=1).mean())


def compute_metrics(gold: np.ndarray, pred: np.ndarray, labels: list[str]) -> tuple[dict, list[dict]]:
    per_label = [
        {
            "label": row["label"],
            "precision": round(row["precision"], 4),
            "recall": round(row["recall"], 4),
            "f1": round(row["f1"], 4),
            "support": row["support"],
        }
        for row in precision_recall_f1_support(gold, pred, labels)
    ]
    global_metrics = {
        "micro_f1": round(micro_f1(gold, pred), 4),
        "macro_f1": round(macro_f1(gold, pred, labels), 4),
        "exact_match": round(exact_match(gold, pred), 4),
    }
    return global_metrics, per_label


def _validate_inputs(gold: np.ndarray, pred: np.ndarray, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    gold, pred = _validate_pair(gold, pred)
    if gold.shape[1] != len(labels):
        raise ValueError(f"labels length must match matrix columns: {len(labels)} != {gold.shape[1]}")
    return gold, pred


def _validate_pair(gold: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gold = np.asarray(gold, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    if gold.ndim != 2 or pred.ndim != 2:
        raise ValueError("gold and pred must be 2D matrices")
    if gold.shape != pred.shape:
        raise ValueError(f"gold and pred must have the same shape: {gold.shape} != {pred.shape}")
    return gold, pred


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0
