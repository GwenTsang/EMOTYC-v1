import json

import numpy as np
import pytest

from emotyc.compare import compare_metrics
from emotyc.encoders import OnnxBackboneEncoder
from emotyc.heads import ClassificationHead


def write_head_config(path, n_labels=2):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "head_type": "camembert_sequence_classification_head",
                "pooling": {"type": "token_index", "axis": 1, "index": 0},
                "layers": [
                    {
                        "type": "linear",
                        "weight": "classifier.dense.weight",
                        "bias": "classifier.dense.bias",
                        "activation": "tanh",
                    },
                    {
                        "type": "linear",
                        "weight": "classifier.out_proj.weight",
                        "bias": "classifier.out_proj.bias",
                        "activation": None,
                    },
                ],
                "output": {"activation_for_probabilities": "sigmoid", "num_labels": n_labels},
            }
        ),
        encoding="utf-8",
    )


def test_classification_head_logits(tmp_path):
    weights_path = tmp_path / "head.npz"
    config_path = tmp_path / "head.json"
    write_head_config(config_path)
    np.savez(
        weights_path,
        **{
            "classifier.dense.weight": np.eye(2, dtype=np.float32),
            "classifier.dense.bias": np.zeros(2, dtype=np.float32),
            "classifier.out_proj.weight": np.eye(2, dtype=np.float32),
            "classifier.out_proj.bias": np.zeros(2, dtype=np.float32),
        },
    )

    head = ClassificationHead.from_files(weights_path, config_path)
    logits = head.logits(np.array([[0.0, 1.0]], dtype=np.float32))

    assert logits.shape == (1, 2)
    assert np.allclose(logits, np.tanh([[0.0, 1.0]]))


def test_invalid_head_schema_fails(tmp_path):
    weights_path = tmp_path / "head.npz"
    config_path = tmp_path / "head.json"
    write_head_config(config_path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["schema_version"] = 2
    config_path.write_text(json.dumps(data), encoding="utf-8")
    np.savez(weights_path)

    with pytest.raises(ValueError, match="Unsupported head.json schema"):
        ClassificationHead.from_files(weights_path, config_path)


def test_onnx_encoder_does_not_add_special_tokens():
    class FakeEncoding:
        def __init__(self, ids=None):
            self.ids = [10, 11] if ids is None else ids

    class FakeTokenizer:
        def __init__(self):
            self.add_special_tokens = None

        def encode(self, text, add_special_tokens):
            return FakeEncoding()

        def encode_batch(self, texts, add_special_tokens):
            self.add_special_tokens = add_special_tokens
            return [FakeEncoding() for _ in texts]

    tokenizer = FakeTokenizer()
    encoder = OnnxBackboneEncoder(session=object(), tokenizer=tokenizer, input_names=set(), pad_id=1)

    inputs = encoder._encode_batch(["before:</s>current: hello</s>after:</s>"])

    assert tokenizer.add_special_tokens is False
    assert inputs["input_ids"].tolist() == [[10, 11]]


def test_onnx_encoder_sorts_by_length_and_restores_original_order():
    class FakeEncoding:
        def __init__(self, ids):
            self.ids = ids

    class FakeTokenizer:
        values = {
            "long": [30, 31, 32],
            "s": [10],
            "mid": [20, 21],
        }

        def encode(self, text, add_special_tokens):
            return FakeEncoding(self.values[text])

        def encode_batch(self, texts, add_special_tokens):
            return [FakeEncoding(self.values[text]) for text in texts]

    class FakeSession:
        def __init__(self):
            self.batch_first_ids = []

        def run(self, _outputs, inputs):
            first_ids = inputs["input_ids"][:, 0].astype(np.float32)
            self.batch_first_ids.extend(first_ids.astype(int).tolist())
            hidden = first_ids.reshape(-1, 1, 1)
            return [hidden]

    session = FakeSession()
    encoder = OnnxBackboneEncoder(session=session, tokenizer=FakeTokenizer(), input_names=set(), pad_id=1)

    features = encoder.encode(["long", "s", "mid"], batch_size=2)

    assert session.batch_first_ids == [30, 10, 20]
    assert features.tolist() == [[30.0], [10.0], [20.0]]


def test_compare_metrics_reports_deltas_and_unique_labels(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(
        json.dumps(
            {
                "global_metrics": {"micro_f1": 0.5, "macro_f1": 0.4, "exact_match": 0.1},
                "per_label": [{"label": "A", "precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}],
                "labels_evaluated": ["A", "OnlyA"],
            }
        ),
        encoding="utf-8",
    )
    b.write_text(
        json.dumps(
            {
                "global_metrics": {"micro_f1": 0.7, "macro_f1": 0.6, "exact_match": 0.2},
                "per_label": [{"label": "A", "precision": 0.6, "recall": 0.7, "f1": 0.8, "support": 3}],
                "labels_evaluated": ["A", "OnlyB"],
            }
        ),
        encoding="utf-8",
    )

    comparison = compare_metrics(a, b)

    assert comparison["global"]["micro_f1"]["delta"] == 0.2
    assert comparison["per_label"]["A"]["f1"]["delta"] == 0.3
    assert comparison["labels_common"] == ["A"]
    assert comparison["labels_only_a"] == ["OnlyA"]
    assert comparison["labels_only_b"] == ["OnlyB"]
