import pandas as pd
import pytest

from emotyc.datasets import load_xlsx


def test_predict_xlsx_accepts_text_only(tmp_path):
    path = tmp_path / "input.xlsx"
    pd.DataFrame({"TEXT": ["a", "b"], "ignored": [1, 2]}).to_excel(path, index=False)

    dataset = load_xlsx(path)

    assert dataset.texts == ["a", "b"]
    assert dataset.gold is None


def test_evaluate_uses_strict_label_intersection(tmp_path):
    path = tmp_path / "gold.xlsx"
    pd.DataFrame({"TEXT": ["a", "b"], "A": [1, 0]}).to_excel(path, index=False)

    dataset = load_xlsx(path, model_labels=["A", "B"], require_gold=True)

    assert dataset.labels_evaluated == ["A"]
    assert dataset.labels_missing == ["B"]
    assert dataset.gold.tolist() == [[1], [0]]


def test_missing_text_fails(tmp_path):
    path = tmp_path / "bad.xlsx"
    pd.DataFrame({"A": [1]}).to_excel(path, index=False)

    with pytest.raises(ValueError, match="TEXT"):
        load_xlsx(path)


def test_no_common_label_fails_for_evaluation(tmp_path):
    path = tmp_path / "bad.xlsx"
    pd.DataFrame({"TEXT": ["a"], "A": [1]}).to_excel(path, index=False)

    with pytest.raises(ValueError, match="no label column"):
        load_xlsx(path, model_labels=["B"], require_gold=True)


def test_non_binary_label_column_fails(tmp_path):
    path = tmp_path / "bad.xlsx"
    pd.DataFrame({"TEXT": ["a", "b"], "A": [1, 2]}).to_excel(path, index=False)

    with pytest.raises(ValueError, match="binary"):
        load_xlsx(path, model_labels=["A"], require_gold=True)
