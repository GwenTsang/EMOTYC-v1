from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class XlsxDataset:
    frame: pd.DataFrame
    texts: list[str]
    labels_evaluated: list[str]
    gold: np.ndarray | None
    labels_present: list[str]
    labels_missing: list[str]


def load_xlsx(path: str | Path, model_labels: list[str] | None = None, require_gold: bool = False) -> XlsxDataset:
    frame = pd.read_excel(path)
    if "TEXT" not in frame.columns:
        raise ValueError("XLSX file must contain a TEXT column")

    texts = frame["TEXT"].fillna("").astype(str).tolist()
    if model_labels is None:
        return XlsxDataset(frame, texts, [], None, [], [])

    labels_present = [label for label in model_labels if label in frame.columns]
    labels_missing = [label for label in model_labels if label not in frame.columns]
    if require_gold and not labels_present:
        raise ValueError("XLSX file has no label column in common with model_config.json")

    gold = None
    if labels_present:
        values = frame[labels_present].fillna(0)
        invalid: list[str] = []
        for label in labels_present:
            unique = set(values[label].dropna().tolist())
            if not unique.issubset({0, 1, 0.0, 1.0, False, True}):
                invalid.append(label)
        if invalid:
            raise ValueError(f"Gold label columns must be binary: {', '.join(invalid)}")
        gold = values.astype(np.int64).to_numpy()

    return XlsxDataset(frame, texts, labels_present, gold, labels_present, labels_missing)
