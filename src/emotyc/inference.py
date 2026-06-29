from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from emotyc.encoders import Encoder


class Head(Protocol):
    def logits(self, features: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class PredictionResult:
    logits: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    labels: list[str]


def run_inference(
    encoder: Encoder,
    head: Head,
    labels: Sequence[str],
    texts: Sequence[str],
    batch_size: int = 32,
    threshold: float = 0.5,
) -> PredictionResult:
    """Run batched inference and apply a global threshold."""
    label_list = list(labels)
    text_list = [str(text) for text in texts]
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    if not text_list:
        empty = np.empty((0, len(label_list)), dtype=np.float32)
        return PredictionResult(
            logits=empty.copy(),
            probabilities=empty.copy(),
            predictions=empty.astype(np.int64),
            labels=label_list,
        )

    features = encoder.encode(text_list, batch_size=batch_size)
    logits = np.asarray(head.logits(features), dtype=np.float32)
    if logits.ndim != 2:
        raise ValueError(f"Head logits must be 2D, got shape {logits.shape}")
    if logits.shape != (len(text_list), len(label_list)):
        raise ValueError(
            "Head logits shape does not match texts and labels: "
            f"got {logits.shape}, expected {(len(text_list), len(label_list))}"
        )
    probabilities = sigmoid(logits)
    predictions = apply_threshold(probabilities, threshold)
    return PredictionResult(logits, probabilities, predictions, label_list)


def apply_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Apply a global threshold to probability scores."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return (np.asarray(probabilities) >= threshold).astype(np.int64)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Compute a numerically stable sigmoid."""
    x = np.asarray(logits, dtype=np.float32)
    out = np.empty_like(x, dtype=np.float32)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out
