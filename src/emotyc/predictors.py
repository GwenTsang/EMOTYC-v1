from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from emotyc.artifacts import ModelBundle
from emotyc.encoders import OnnxBackboneEncoder
from emotyc.heads import ClassificationHead
from emotyc.inference import PredictionResult, run_inference


@dataclass
class Predictor:
    encoder: OnnxBackboneEncoder
    head: ClassificationHead
    labels: list[str]

    @classmethod
    def from_bundle(cls, bundle: ModelBundle) -> "Predictor":
        labels = labels_from_model_config(bundle.model_config)
        head = ClassificationHead.from_files(bundle.head_weights, bundle.head_config)
        if head.n_labels != len(labels):
            raise ValueError("head.json n_labels does not match model_config.json labels")
        encoder = OnnxBackboneEncoder.from_files(str(bundle.backbone), str(bundle.tokenizer))
        return cls(encoder=encoder, head=head, labels=labels)

    def predict(
        self,
        texts: list[str],
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


def labels_from_model_config(path: str | Path) -> list[str]:
    import json

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = config.get("labels")
    if isinstance(labels, list) and all(isinstance(label, str) for label in labels):
        return labels
    id2label = config.get("id2label")
    if isinstance(id2label, dict):
        return [id2label[str(index)] for index in range(len(id2label))]
    raise ValueError("model_config.json must contain labels or id2label")
