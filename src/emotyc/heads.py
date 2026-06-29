from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ClassificationHead:
    dense_weight: np.ndarray
    dense_bias: np.ndarray
    out_proj_weight: np.ndarray
    out_proj_bias: np.ndarray
    n_labels: int

    @classmethod
    def from_files(cls, weights_path: str | Path, config_path: str | Path) -> "ClassificationHead":
        config = load_head_config(config_path)
        weights = np.load(weights_path)
        required = (
            "classifier.dense.weight",
            "classifier.dense.bias",
            "classifier.out_proj.weight",
            "classifier.out_proj.bias",
        )
        missing = [name for name in required if name not in weights]
        if missing:
            raise ValueError(f"head.npz is missing weights: {', '.join(missing)}")
        n_labels = int(config["output"]["num_labels"])
        head = cls(
            dense_weight=np.asarray(weights["classifier.dense.weight"], dtype=np.float32),
            dense_bias=np.asarray(weights["classifier.dense.bias"], dtype=np.float32),
            out_proj_weight=np.asarray(weights["classifier.out_proj.weight"], dtype=np.float32),
            out_proj_bias=np.asarray(weights["classifier.out_proj.bias"], dtype=np.float32),
            n_labels=n_labels,
        )
        if head.out_proj_bias.shape[0] != n_labels:
            raise ValueError("head.json n_labels does not match out_proj.bias")
        head._validate_shapes()
        return head

    def logits(self, cls_embeddings: np.ndarray) -> np.ndarray:
        hidden = np.tanh(cls_embeddings @ self.dense_weight.T + self.dense_bias)
        return hidden @ self.out_proj_weight.T + self.out_proj_bias

    def _validate_shapes(self) -> None:
        if self.dense_weight.ndim != 2:
            raise ValueError("classifier.dense.weight must be a 2D matrix")
        if self.dense_bias.shape != (self.dense_weight.shape[0],):
            raise ValueError("classifier.dense.bias shape does not match classifier.dense.weight")
        if self.out_proj_weight.ndim != 2:
            raise ValueError("classifier.out_proj.weight must be a 2D matrix")
        if self.out_proj_weight.shape != (self.n_labels, self.dense_weight.shape[0]):
            raise ValueError("classifier.out_proj.weight shape does not match dense output and labels")
        if self.out_proj_bias.shape != (self.n_labels,):
            raise ValueError("classifier.out_proj.bias shape does not match labels")


def load_head_config(path: str | Path) -> dict:
    """Load and validate a head.json file."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_head_config(config)
    return config


def validate_head_config(config: object) -> None:
    """Validate the supported v1 head schema."""
    if not isinstance(config, dict):
        raise ValueError("Unsupported head.json schema: root must be an object")
    mismatches = []
    if config.get("schema_version") != 1:
        mismatches.append("schema_version must be 1")
    if config.get("head_type") != "camembert_sequence_classification_head":
        mismatches.append("head_type must be camembert_sequence_classification_head")
    pooling = config.get("pooling", {})
    if pooling.get("type") != "token_index" or pooling.get("axis") != 1 or pooling.get("index") != 0:
        mismatches.append("pooling must select last_hidden_state[:, 0, :]")
    layers = config.get("layers")
    if not isinstance(layers, list) or len(layers) != 2:
        mismatches.append("layers must contain dense and out_proj")
    else:
        dense, out_proj = layers
        if dense.get("type") != "linear" or dense.get("activation") != "tanh":
            mismatches.append("dense layer must be linear with tanh activation")
        if dense.get("weight") != "classifier.dense.weight" or dense.get("bias") != "classifier.dense.bias":
            mismatches.append("dense layer weight names are unsupported")
        if out_proj.get("type") != "linear" or out_proj.get("activation") is not None:
            mismatches.append("out_proj layer must be linear without activation")
        if (
            out_proj.get("weight") != "classifier.out_proj.weight"
            or out_proj.get("bias") != "classifier.out_proj.bias"
        ):
            mismatches.append("out_proj layer weight names are unsupported")
    output = config.get("output", {})
    if output.get("activation_for_probabilities") != "sigmoid":
        mismatches.append("output activation_for_probabilities must be sigmoid")
    if not isinstance(output.get("num_labels"), int):
        mismatches.append("output.num_labels is missing")
    if mismatches:
        raise ValueError("Unsupported head.json schema: " + "; ".join(mismatches))


def _validate_head_config(config: object) -> None:
    validate_head_config(config)
