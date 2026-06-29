import json

import pytest

from emotyc.artifacts import REQUIRED_MODEL_FILES, bundle_from_directory
from emotyc.heads import load_head_config, validate_head_config
from emotyc.registry import (
    dataset_aliases,
    model_aliases,
    resolve_dataset_entry,
    resolve_model_repo,
)


def valid_head_config(n_labels=2):
    return {
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


def test_registry_known_aliases():
    assert "emotyc_1" in model_aliases()
    assert "emotyc_2" in model_aliases()
    assert resolve_model_repo("emotyc_1") == "GwendalTsang/EMOTYC_1"
    assert "ttk" in dataset_aliases()
    assert "CyberAgg" in dataset_aliases()
    assert resolve_dataset_entry("ttk") == {
        "repo_id": "GwendalTsang/TTK",
        "filename": "emotexttokids_gold_flat.xlsx",
    }
    assert resolve_dataset_entry("CyberAgg") == {
        "repo_id": "GwendalTsang/CyberAggAdo",
        "filename": "CyberAdoAgg_gold_global_total.xlsx",
    }


def test_registry_unknown_alias_error_lists_available_aliases():
    with pytest.raises(ValueError) as exc_info:
        resolve_model_repo("missing")

    message = str(exc_info.value)
    assert "Unknown model alias" in message
    assert "emotyc_1" in message
    assert "emotyc_2" in message


def test_registry_unknown_dataset_alias_error_lists_available_aliases():
    with pytest.raises(ValueError) as exc_info:
        resolve_dataset_entry("missing")

    message = str(exc_info.value)
    assert "Unknown dataset alias" in message
    assert "CyberAgg" in message
    assert "ttk" in message


def test_bundle_from_directory_accepts_complete_local_bundle(tmp_path):
    for filename in REQUIRED_MODEL_FILES:
        (tmp_path / filename).write_text("", encoding="utf-8")

    bundle = bundle_from_directory(tmp_path)

    assert bundle.root == tmp_path
    assert bundle.backbone == tmp_path / "backbone.onnx"
    assert bundle.head_weights == tmp_path / "head.npz"
    assert bundle.head_config == tmp_path / "head.json"
    assert bundle.tokenizer == tmp_path / "tokenizer.json"
    assert bundle.model_config == tmp_path / "model_config.json"


def test_bundle_from_directory_rejects_incomplete_bundle(tmp_path):
    (tmp_path / "head.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Incomplete model bundle"):
        bundle_from_directory(tmp_path)


def test_load_head_config_accepts_supported_schema(tmp_path):
    path = tmp_path / "head.json"
    path.write_text(json.dumps(valid_head_config()), encoding="utf-8")

    assert load_head_config(path)["output"]["num_labels"] == 2


def test_validate_head_config_rejects_unsupported_schema():
    config = valid_head_config()
    config["output"]["activation_for_probabilities"] = "softmax"

    with pytest.raises(ValueError, match="Unsupported head.json schema"):
        validate_head_config(config)
