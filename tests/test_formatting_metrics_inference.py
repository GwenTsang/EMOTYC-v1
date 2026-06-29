import numpy as np

from emotyc.formatting import apply_template
from emotyc.metrics import (
    compute_metrics,
    exact_match,
    macro_f1,
    micro_f1,
    precision_recall_f1_support,
)


def test_formatting_templates_raw_and_bca():
    assert apply_template(["hello"], "raw") == ["hello"]
    assert apply_template(["hello"], "bca") == ["before:</s>current: hello</s>after:</s>"]


def test_formatting_bca_with_context_matches_eval_emotyc_use_context():
    texts = apply_template(["a", "b", "c"], "bca", use_context=True)

    assert texts == [
        "before:</s></s>current: a</s>after:b</s>",
        "before:a</s>current: b</s>after:c</s>",
        "before:b</s>current: c</s>after:</s></s>",
    ]


def test_metrics_numpy_multilabel_contract():
    gold = np.asarray(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.int64,
    )
    pred = np.asarray(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.int64,
    )
    labels = ["A", "B", "C"]

    per_label = precision_recall_f1_support(gold, pred, labels)
    assert per_label[0] == {
        "label": "A",
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "support": 2,
    }
    assert per_label[1] == {
        "label": "B",
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "support": 2,
    }
    assert per_label[2] == {
        "label": "C",
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 1,
    }
    assert np.isclose(micro_f1(gold, pred), 2 / 3)
    assert np.isclose(macro_f1(gold, pred, labels), 0.5)
    assert exact_match(gold, pred) == 0.25

    global_metrics, rounded_per_label = compute_metrics(gold, pred, labels)
    assert global_metrics == {"micro_f1": 0.6667, "macro_f1": 0.5, "exact_match": 0.25}
    assert rounded_per_label[0]["support"] == 2
