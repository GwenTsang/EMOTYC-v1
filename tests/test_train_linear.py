import numpy as np
import pandas as pd
import pytest

from emotyc.cli.train_linear import parse_labels
from emotyc.linear import load_linear_model, resolve_backbone_files, train_linear_model


def test_parse_labels_accepts_spaces_and_commas():
    assert parse_labels(["A", "B,C"]) == ["A", "B", "C"]


def test_parse_labels_rejects_duplicates():
    with pytest.raises(ValueError, match="dupliques"):
        parse_labels(["A", "B", "A"])


def test_train_linear_tfidf_roundtrip(tmp_path):
    pytest.importorskip("sklearn")
    texts = [
        "alpha positif",
        "alpha excellent",
        "beta negatif",
        "beta mauvais",
    ]
    labels = ["A", "B"]
    gold = np.asarray(
        [
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
        ],
        dtype=np.int64,
    )

    classifier = train_linear_model(
        texts=texts,
        gold=gold,
        labels=labels,
        encoder_type="tfidf",
        tfidf_ngram_range=(1, 1),
        max_iter=2000,
    )
    out_dir = tmp_path / "linear"
    classifier.save(out_dir)

    loaded = load_linear_model(out_dir)
    result = loaded.predict(texts, threshold=0.5)

    assert loaded.labels == labels
    assert result.predictions.shape == (4, 2)
    assert result.probabilities.shape == (4, 2)
    assert (out_dir / "linear_config.json").is_file()
    assert (out_dir / "linear_model.pkl").is_file()


def test_train_linear_requires_declared_labels_in_xlsx(tmp_path):
    path = tmp_path / "train.xlsx"
    pd.DataFrame({"TEXT": ["a", "b"], "A": [0, 1]}).to_excel(path, index=False)

    from emotyc.cli.train_linear import build_parser, train

    args = build_parser().parse_args(
        [
            "train",
            "--data",
            str(path),
            "--labels",
            "A",
            "B",
            "--out-dir",
            str(tmp_path / "model"),
        ]
    )
    with pytest.raises(ValueError, match="Colonnes labels manquantes"):
        train(args)


def test_resolve_backbone_files_accepts_backbone_only_directory(tmp_path):
    (tmp_path / "model.onnx").write_text("", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

    backbone, tokenizer = resolve_backbone_files(str(tmp_path))

    assert backbone == tmp_path / "model.onnx"
    assert tokenizer == tmp_path / "tokenizer.json"
