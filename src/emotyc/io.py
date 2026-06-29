from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    """Write a JSON file with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def export_predictions_xlsx(
    path: str | Path,
    texts: Sequence[str],
    labels: Sequence[str],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    gold: np.ndarray | None = None,
) -> None:
    """Export predictions, probabilities and optional gold labels to XLSX."""
    label_list = list(labels)
    text_list = [str(text) for text in texts]
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)
    expected_shape = (len(text_list), len(label_list))
    if predictions.shape != expected_shape:
        raise ValueError(f"predictions shape must be {expected_shape}, got {predictions.shape}")
    if probabilities.shape != expected_shape:
        raise ValueError(f"probabilities shape must be {expected_shape}, got {probabilities.shape}")

    gold_array = None if gold is None else np.asarray(gold)
    if gold_array is not None and gold_array.shape != expected_shape:
        raise ValueError(f"gold shape must be {expected_shape}, got {gold_array.shape}")

    frame = pd.DataFrame({"TEXT": text_list})
    for label_index, label in enumerate(label_list):
        if gold_array is not None:
            frame[label] = gold_array[:, label_index]
        frame[f"pred_{label}"] = predictions[:, label_index]
        frame[f"proba_{label}"] = probabilities[:, label_index]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(path, index=False)
