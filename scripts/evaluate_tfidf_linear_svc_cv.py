from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from emotyc.artifacts import resolve_model_bundle
from emotyc.data_sources import resolve_data, resolve_dataset
from emotyc.datasets import load_xlsx
from emotyc.formatting import apply_template
from emotyc.io import export_predictions_xlsx, write_json
from emotyc.metrics import compute_metrics
from emotyc.predictors import labels_from_model_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluer TF-IDF + LinearSVC en validation croisee multi-label."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="Alias dataset connu.")
    source.add_argument("--data", help="Fichier XLSX local ou alias dataset connu.")
    parser.add_argument("--labels-model", default="emotyc_1", help="Modele dont reprendre l'ordre des labels.")
    parser.add_argument("--template", choices=("raw", "bca"), default="raw")
    parser.add_argument(
        "--use-context",
        action="store_true",
        help="Utiliser les lignes voisines i-1 et i+1 dans le template bca.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Nombre de folds KFold.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-features", type=int)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--no-lowercase", action="store_true")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--class-weight", choices=("none", "balanced"), default="balanced")
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--save-predictions", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


def run(args: argparse.Namespace) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import KFold
    from sklearn.svm import LinearSVC

    labels = labels_from_model_config(resolve_model_bundle(args.labels_model).model_config)
    data_path = resolve_dataset(args.dataset) if args.dataset else resolve_data(args.data)
    dataset = load_xlsx(data_path, model_labels=labels, require_gold=True)
    if dataset.gold is None:
        raise ValueError("Le fichier XLSX ne contient aucune colonne label exploitable.")

    texts = apply_template(dataset.texts, args.template, use_context=args.use_context)
    gold = np.asarray(dataset.gold, dtype=np.int64)
    n_samples, n_labels = gold.shape
    if args.folds < 2 or args.folds > n_samples:
        raise ValueError("--folds must be between 2 and the number of samples")

    predictions = np.zeros_like(gold, dtype=np.int64)
    scores = np.zeros((n_samples, n_labels), dtype=np.float32)
    fold_metrics: list[dict] = []
    label_training_modes: list[dict] = []
    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)

    for fold_index, (train_index, test_index) in enumerate(kfold.split(texts), start=1):
        train_texts = [texts[index] for index in train_index]
        test_texts = [texts[index] for index in test_index]
        vectorizer = TfidfVectorizer(
            max_features=args.max_features,
            ngram_range=(args.ngram_min, args.ngram_max),
            lowercase=not args.no_lowercase,
        )
        train_features = vectorizer.fit_transform(train_texts)
        test_features = vectorizer.transform(test_texts)
        train_gold = gold[train_index]
        test_gold = gold[test_index]
        fold_pred = np.zeros_like(test_gold, dtype=np.int64)

        for label_index, label in enumerate(dataset.labels_evaluated):
            target = train_gold[:, label_index]
            unique = sorted(set(target.tolist()))
            if len(unique) < 2:
                constant = int(unique[0])
                fold_scores = np.full(len(test_index), float(constant), dtype=np.float32)
                fold_label_pred = np.full(len(test_index), constant, dtype=np.int64)
                mode = f"constant_{constant}"
            else:
                classifier = LinearSVC(
                    C=args.c,
                    class_weight=None if args.class_weight == "none" else args.class_weight,
                    max_iter=args.max_iter,
                    random_state=args.random_state,
                )
                classifier.fit(train_features, target)
                decision = np.asarray(classifier.decision_function(test_features), dtype=np.float32)
                fold_scores = sigmoid(decision)
                fold_label_pred = (decision >= 0).astype(np.int64)
                mode = "linear_svc"
            scores[test_index, label_index] = fold_scores
            predictions[test_index, label_index] = fold_label_pred
            fold_pred[:, label_index] = fold_label_pred
            label_training_modes.append(
                {
                    "fold": fold_index,
                    "label": label,
                    "mode": mode,
                    "train_positive": int(target.sum()),
                    "train_negative": int(len(target) - target.sum()),
                }
            )

        fold_global, _ = compute_metrics(test_gold, fold_pred, dataset.labels_evaluated)
        fold_metrics.append(
            {
                "fold": fold_index,
                "n_train": int(len(train_index)),
                "n_test": int(len(test_index)),
                **fold_global,
            }
        )
        print(
            f"fold {fold_index}/{args.folds}: "
            f"micro-F1={fold_global['micro_f1']:.4f} "
            f"macro-F1={fold_global['macro_f1']:.4f} "
            f"exact_match={fold_global['exact_match']:.4f}"
        )

    global_metrics, per_label = compute_metrics(gold, predictions, dataset.labels_evaluated)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": args.dataset,
        "data": args.data,
        "data_path": str(data_path),
        "labels_model": args.labels_model,
        "labels": dataset.labels_evaluated,
        "template": args.template,
        "use_context": args.use_context,
        "folds": args.folds,
        "random_state": args.random_state,
        "encoder": {
            "type": "tfidf",
            "max_features": args.max_features,
            "ngram_range": [args.ngram_min, args.ngram_max],
            "lowercase": not args.no_lowercase,
        },
        "head": {
            "type": "linear_svc",
            "c": args.c,
            "class_weight": None if args.class_weight == "none" else args.class_weight,
            "max_iter": args.max_iter,
        },
    }
    write_json(out_dir / "config.json", config)
    write_json(
        out_dir / "metrics.json",
        {
            "global_metrics": global_metrics,
            "per_label": per_label,
            "labels_evaluated": dataset.labels_evaluated,
            "n_samples": len(dataset.texts),
            "fold_metrics": fold_metrics,
            "label_training_modes": label_training_modes,
        },
    )
    if args.save_predictions:
        export_predictions_xlsx(
            out_dir / "predictions.xlsx",
            texts=dataset.texts,
            labels=dataset.labels_evaluated,
            predictions=predictions,
            probabilities=scores,
            gold=gold,
        )
    print("Metriques globales")
    print(f"micro-F1: {global_metrics['micro_f1']:.4f}")
    print(f"macro-F1: {global_metrics['macro_f1']:.4f}")
    print(f"exact match: {global_metrics['exact_match']:.4f}")


def sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    out = np.empty_like(x, dtype=np.float32)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


if __name__ == "__main__":
    main()
