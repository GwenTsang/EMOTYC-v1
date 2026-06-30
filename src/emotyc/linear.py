from __future__ import annotations

import pickle
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from emotyc.artifacts import resolve_model_bundle
from emotyc.encoders import Encoder, OnnxBackboneEncoder
from emotyc.inference import PredictionResult, run_inference
from emotyc.io import read_json, write_json

LINEAR_CONFIG_FILENAME = "linear_config.json"
LINEAR_MODEL_FILENAME = "linear_model.pkl"
LINEAR_SCHEMA_VERSION = 1
LINEAR_MODEL_TYPE = "linear_svc_multilabel"


@dataclass
class TfidfEncoder:
    vectorizer: object

    def encode(self, texts: list[str], batch_size: int) -> object:
        del batch_size
        return self.vectorizer.transform(texts)


@dataclass
class LinearSVCHead:
    classifiers: list[object]

    def logits(self, features: object) -> np.ndarray:
        columns = []
        for classifier in self.classifiers:
            scores = np.asarray(classifier.decision_function(features), dtype=np.float32).reshape(-1)
            columns.append(scores)
        if not columns:
            return np.empty((0, 0), dtype=np.float32)
        return np.column_stack(columns).astype(np.float32, copy=False)


@dataclass
class LinearTextClassifier:
    encoder: Encoder
    head: LinearSVCHead
    labels: list[str]
    config: dict[str, Any]
    source_files: dict[str, Path] = field(default_factory=dict)

    def predict(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        threshold: float = 0.5,
    ) -> PredictionResult:
        return run_inference(
            encoder=self.encoder,
            head=self.head,
            labels=self.labels,
            texts=texts,
            batch_size=batch_size,
            threshold=threshold,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config = dict(self.config)
        config["schema_version"] = LINEAR_SCHEMA_VERSION
        config["model_type"] = LINEAR_MODEL_TYPE
        config["labels"] = self.labels

        payload: dict[str, object] = {"classifiers": self.head.classifiers}
        encoder_config = config.get("encoder", {})
        if encoder_config.get("type") == "tfidf":
            if not isinstance(self.encoder, TfidfEncoder):
                raise ValueError("TF-IDF artifacts require a TfidfEncoder")
            payload["vectorizer"] = self.encoder.vectorizer
        elif encoder_config.get("type") == "onnx":
            self._copy_onnx_files(path)
        else:
            raise ValueError("Unsupported linear encoder type")

        with (path / LINEAR_MODEL_FILENAME).open("wb") as handle:
            pickle.dump(payload, handle)
        write_json(path / LINEAR_CONFIG_FILENAME, config)

    def _copy_onnx_files(self, path: Path) -> None:
        for key in ("backbone", "tokenizer"):
            source = self.source_files.get(key)
            if source is None:
                raise ValueError(f"Missing ONNX source file for {key}")
            filename = "backbone.onnx" if key == "backbone" else "tokenizer.json"
            destination = path / filename
            if Path(source).resolve() != destination.resolve():
                shutil.copy2(source, destination)


def train_linear_model(
    texts: Sequence[str],
    gold: np.ndarray,
    labels: Sequence[str],
    encoder_type: str = "tfidf",
    batch_size: int = 32,
    tfidf_max_features: int | None = None,
    tfidf_ngram_range: tuple[int, int] = (1, 2),
    tfidf_lowercase: bool = True,
    c: float = 1.0,
    class_weight: str | None = None,
    max_iter: int = 1000,
    backbone_model: str | None = None,
) -> LinearTextClassifier:
    """Train one LinearSVC classifier per label."""
    text_list = [str(text) for text in texts]
    label_list = list(labels)
    gold = _validate_training_inputs(gold, label_list, len(text_list))

    if encoder_type == "tfidf":
        encoder, features, encoder_config = _fit_tfidf_encoder(
            text_list,
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
            lowercase=tfidf_lowercase,
        )
        source_files: dict[str, Path] = {}
    elif encoder_type == "onnx":
        encoder, features, encoder_config, source_files = _build_onnx_features(
            text_list,
            batch_size=batch_size,
            backbone_model=backbone_model,
        )
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")

    classifiers = fit_linear_svc_head(
        features=features,
        gold=gold,
        labels=label_list,
        c=c,
        class_weight=class_weight,
        max_iter=max_iter,
    )
    config = {
        "schema_version": LINEAR_SCHEMA_VERSION,
        "model_type": LINEAR_MODEL_TYPE,
        "labels": label_list,
        "encoder": encoder_config,
        "head": {
            "type": "linear_svc",
            "c": c,
            "class_weight": class_weight,
            "max_iter": max_iter,
        },
    }
    return LinearTextClassifier(
        encoder=encoder,
        head=LinearSVCHead(classifiers),
        labels=label_list,
        config=config,
        source_files=source_files,
    )


def fit_linear_svc_head(
    features: object,
    gold: np.ndarray,
    labels: Sequence[str],
    c: float = 1.0,
    class_weight: str | None = None,
    max_iter: int = 1000,
) -> list[object]:
    """Fit one LinearSVC estimator per label."""
    classifiers = []
    for label_index, label in enumerate(labels):
        target = gold[:, label_index]
        if len(set(target.tolist())) < 2:
            raise ValueError(f"Label '{label}' must contain both classes 0 and 1")
        classifier = LinearSVC(C=c, class_weight=class_weight, max_iter=max_iter)
        classifier.fit(features, target)
        classifiers.append(classifier)
    return classifiers


def load_linear_model(path: str | Path) -> LinearTextClassifier:
    """Load a train_linear.py artifact."""
    path = Path(path)
    config = read_json(path / LINEAR_CONFIG_FILENAME)
    _validate_linear_config(config)
    with (path / LINEAR_MODEL_FILENAME).open("rb") as handle:
        payload = pickle.load(handle)

    encoder_config = config["encoder"]
    if encoder_config["type"] == "tfidf":
        vectorizer = payload.get("vectorizer")
        if vectorizer is None:
            raise ValueError("linear_model.pkl is missing the TF-IDF vectorizer")
        encoder: Encoder = TfidfEncoder(vectorizer)
    elif encoder_config["type"] == "onnx":
        encoder = OnnxBackboneEncoder.from_files(
            str(path / encoder_config["backbone"]),
            str(path / encoder_config["tokenizer"]),
        )
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_config['type']}")

    classifiers = payload.get("classifiers")
    if not isinstance(classifiers, list) or len(classifiers) != len(config["labels"]):
        raise ValueError("linear_model.pkl classifiers do not match linear_config.json labels")
    return LinearTextClassifier(
        encoder=encoder,
        head=LinearSVCHead(classifiers),
        labels=list(config["labels"]),
        config=config,
    )


def _fit_tfidf_encoder(
    texts: list[str],
    max_features: int | None,
    ngram_range: tuple[int, int],
    lowercase: bool,
) -> tuple[TfidfEncoder, object, dict[str, Any]]:
    if ngram_range[0] <= 0 or ngram_range[0] > ngram_range[1]:
        raise ValueError("TF-IDF ngram range is invalid")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=lowercase,
    )
    features = vectorizer.fit_transform(texts)
    encoder = TfidfEncoder(vectorizer)
    config = {
        "type": "tfidf",
        "max_features": max_features,
        "ngram_range": list(ngram_range),
        "lowercase": lowercase,
    }
    return encoder, features, config


def _build_onnx_features(
    texts: list[str],
    batch_size: int,
    backbone_model: str | None,
) -> tuple[OnnxBackboneEncoder, np.ndarray, dict[str, Any], dict[str, Path]]:
    if backbone_model is None:
        raise ValueError("--backbone-model is required when --encoder onnx is used")
    backbone_path, tokenizer_path = resolve_backbone_files(backbone_model)
    encoder = OnnxBackboneEncoder.from_files(str(backbone_path), str(tokenizer_path))
    features = encoder.encode(texts, batch_size=batch_size)
    config = {
        "type": "onnx",
        "source_model": backbone_model,
        "backbone": "backbone.onnx",
        "tokenizer": "tokenizer.json",
    }
    return encoder, features, config, {"backbone": backbone_path, "tokenizer": tokenizer_path}


def resolve_backbone_files(backbone_model: str) -> tuple[Path, Path]:
    """Resolve a complete EMOTYC bundle or a backbone-only ONNX directory."""
    path = Path(backbone_model).expanduser()
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Backbone path is not a directory: {path}")
        tokenizer = path / "tokenizer.json"
        if not tokenizer.is_file():
            raise ValueError(f"Backbone directory is missing tokenizer.json: {path}")
        for filename in ("backbone.onnx", "model.onnx"):
            backbone = path / filename
            if backbone.is_file():
                return backbone, tokenizer
        raise ValueError(f"Backbone directory is missing backbone.onnx or model.onnx: {path}")

    bundle = resolve_model_bundle(backbone_model)
    return bundle.backbone, bundle.tokenizer


def _validate_training_inputs(
    gold: np.ndarray,
    labels: list[str],
    n_texts: int,
) -> np.ndarray:
    if not labels:
        raise ValueError("At least one label is required")
    gold = np.asarray(gold, dtype=np.int64)
    if gold.ndim != 2:
        raise ValueError("gold must be a 2D matrix")
    expected_shape = (n_texts, len(labels))
    if gold.shape != expected_shape:
        raise ValueError(f"gold shape must be {expected_shape}, got {gold.shape}")
    return gold


def _validate_linear_config(config: object) -> None:
    if not isinstance(config, dict):
        raise ValueError("linear_config.json must be an object")
    if config.get("schema_version") != LINEAR_SCHEMA_VERSION:
        raise ValueError("Unsupported linear_config.json schema_version")
    if config.get("model_type") != LINEAR_MODEL_TYPE:
        raise ValueError("Unsupported linear_config.json model_type")
    labels = config.get("labels")
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) for label in labels):
        raise ValueError("linear_config.json labels must be a non-empty list of strings")
    encoder = config.get("encoder")
    if not isinstance(encoder, dict) or encoder.get("type") not in {"tfidf", "onnx"}:
        raise ValueError("linear_config.json encoder is invalid")
