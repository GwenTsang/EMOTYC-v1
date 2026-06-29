from __future__ import annotations

import argparse
from pathlib import Path

from emotyc.datasets import load_xlsx
from emotyc.formatting import apply_template
from emotyc.io import export_predictions_xlsx, write_json
from emotyc.linear import load_linear_model, train_linear_model
from emotyc.metrics import compute_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrainer ou evaluer un modele lineaire EMOTYC.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Entrainer un modele lineaire.")
    train.add_argument("--data", required=True, help="Fichier XLSX annote.")
    train.add_argument("--labels", nargs="+", required=True, help="Labels a entrainer, separes par espace ou virgule.")
    train.add_argument("--out-dir", required=True, help="Dossier de sortie du modele lineaire.")
    train.add_argument("--template", choices=("raw", "bca"), default="raw")
    train.add_argument(
        "--use-context",
        action="store_true",
        help="Utiliser les lignes voisines i-1 et i+1 dans le template bca.",
    )
    train.add_argument("--encoder", choices=("tfidf", "onnx"), default="tfidf")
    train.add_argument("--backbone-model", help="Bundle modele utilise pour extraire des features ONNX.")
    train.add_argument("--batch-size", type=int, default=32, help="Taille de batch pour l'encodeur.")
    train.add_argument("--threshold", type=float, default=0.5, help="Seuil global pour les metriques train.")
    train.add_argument("--max-features", type=int, help="Nombre maximal de features TF-IDF.")
    train.add_argument("--ngram-min", type=int, default=1, help="Borne basse des n-grams TF-IDF.")
    train.add_argument("--ngram-max", type=int, default=2, help="Borne haute des n-grams TF-IDF.")
    train.add_argument("--no-lowercase", action="store_true", help="Desactiver la normalisation lowercase TF-IDF.")
    train.add_argument("--c", type=float, default=1.0, help="Parametre C de LinearSVC.")
    train.add_argument("--class-weight", choices=("none", "balanced"), default="none")
    train.add_argument("--max-iter", type=int, default=1000, help="Nombre maximal d'iterations LinearSVC.")
    train.add_argument("--save-predictions", action="store_true", help="Sauver train_predictions.xlsx.")

    evaluate = subparsers.add_parser("evaluate", help="Evaluer un modele lineaire sauvegarde.")
    evaluate.add_argument("--model", required=True, help="Dossier cree par train_linear.py train.")
    evaluate.add_argument("--data", required=True, help="Fichier XLSX annote.")
    evaluate.add_argument("--template", choices=("raw", "bca"), default="raw")
    evaluate.add_argument(
        "--use-context",
        action="store_true",
        help="Utiliser les lignes voisines i-1 et i+1 dans le template bca.",
    )
    evaluate.add_argument("--threshold", type=float, default=0.5, help="Seuil global de prediction.")
    evaluate.add_argument("--batch-size", type=int, default=32, help="Taille de batch.")
    evaluate.add_argument("--out-dir", default=".", help="Dossier de sortie.")
    evaluate.add_argument("--save-config", action="store_true", help="Sauver config.json.")
    evaluate.add_argument("--save-metrics", action="store_true", help="Sauver metrics.json.")
    evaluate.add_argument("--save-predictions", action="store_true", help="Sauver predictions.xlsx.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        raise ValueError(f"Commande inconnue: {args.command}")


def train(args: argparse.Namespace) -> None:
    labels = parse_labels(args.labels)
    dataset = load_xlsx(args.data, model_labels=labels, require_gold=True)
    if dataset.labels_missing:
        raise ValueError("Colonnes labels manquantes: " + ", ".join(dataset.labels_missing))
    if dataset.gold is None:
        raise ValueError("Le fichier XLSX ne contient aucune colonne label exploitable.")

    texts = apply_template(dataset.texts, args.template, use_context=args.use_context)
    classifier = train_linear_model(
        texts=texts,
        gold=dataset.gold,
        labels=labels,
        encoder_type=args.encoder,
        batch_size=args.batch_size,
        tfidf_max_features=args.max_features,
        tfidf_ngram_range=(args.ngram_min, args.ngram_max),
        tfidf_lowercase=not args.no_lowercase,
        c=args.c,
        class_weight=None if args.class_weight == "none" else args.class_weight,
        max_iter=args.max_iter,
        backbone_model=args.backbone_model,
    )
    out_dir = Path(args.out_dir)
    classifier.save(out_dir)

    result = classifier.predict(texts, batch_size=args.batch_size, threshold=args.threshold)
    global_metrics, per_label = compute_metrics(dataset.gold, result.predictions, labels)
    write_json(
        out_dir / "training_config.json",
        {
            "data": args.data,
            "labels": labels,
            "template": args.template,
            "use_context": args.use_context,
            "encoder": args.encoder,
            "backbone_model": args.backbone_model,
            "threshold": args.threshold,
            "batch_size": args.batch_size,
        },
    )
    write_json(
        out_dir / "train_metrics.json",
        {
            "global_metrics": global_metrics,
            "per_label": per_label,
            "labels_evaluated": labels,
            "n_samples": len(dataset.texts),
            "threshold": args.threshold,
        },
    )
    if args.save_predictions:
        export_predictions_xlsx(
            out_dir / "train_predictions.xlsx",
            texts=dataset.texts,
            labels=labels,
            predictions=result.predictions,
            probabilities=result.probabilities,
            gold=dataset.gold,
        )

    print("Modele lineaire sauvegarde dans " + str(out_dir))
    print_metrics(global_metrics, per_label)


def evaluate(args: argparse.Namespace) -> None:
    classifier = load_linear_model(args.model)
    dataset = load_xlsx(args.data, model_labels=classifier.labels, require_gold=True)
    if dataset.gold is None:
        raise ValueError("Le fichier XLSX ne contient aucune colonne label exploitable.")

    texts = apply_template(dataset.texts, args.template, use_context=args.use_context)
    result = classifier.predict(texts, batch_size=args.batch_size, threshold=args.threshold)
    indices = [classifier.labels.index(label) for label in dataset.labels_evaluated]
    predictions = result.predictions[:, indices]
    probabilities = result.probabilities[:, indices]
    global_metrics, per_label = compute_metrics(dataset.gold, predictions, dataset.labels_evaluated)
    print_metrics(global_metrics, per_label)
    if dataset.labels_missing:
        print("")
        print("Labels absents: " + ", ".join(dataset.labels_missing))

    if args.save_config or args.save_metrics or args.save_predictions:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.save_config:
            write_json(
                out_dir / "config.json",
                {
                    "model": args.model,
                    "data": args.data,
                    "template": args.template,
                    "use_context": args.use_context,
                    "threshold": args.threshold,
                    "batch_size": args.batch_size,
                },
            )
        if args.save_metrics:
            write_json(
                out_dir / "metrics.json",
                {
                    "global_metrics": global_metrics,
                    "per_label": per_label,
                    "labels_evaluated": dataset.labels_evaluated,
                    "n_samples": len(dataset.texts),
                    "threshold": args.threshold,
                },
            )
        if args.save_predictions:
            export_predictions_xlsx(
                out_dir / "predictions.xlsx",
                texts=dataset.texts,
                labels=dataset.labels_evaluated,
                predictions=predictions,
                probabilities=probabilities,
                gold=dataset.gold,
            )


def parse_labels(values: list[str]) -> list[str]:
    labels = []
    for value in values:
        labels.extend(label.strip() for label in value.split(",") if label.strip())
    if not labels:
        raise ValueError("Au moins un label doit etre fourni.")
    seen = set()
    duplicates = []
    for label in labels:
        if label in seen:
            duplicates.append(label)
        seen.add(label)
    if duplicates:
        raise ValueError("Labels dupliques: " + ", ".join(sorted(set(duplicates))))
    return labels


def print_metrics(global_metrics: dict, per_label: list[dict]) -> None:
    print("Metriques globales")
    print(f"micro-F1: {global_metrics['micro_f1']:.4f}")
    print(f"macro-F1: {global_metrics['macro_f1']:.4f}")
    print(f"exact match: {global_metrics['exact_match']:.4f}")
    print("")
    print("Metriques par label")
    for row in per_label:
        print(
            f"{row['label']}\tprecision={row['precision']:.4f}\t"
            f"recall={row['recall']:.4f}\tf1={row['f1']:.4f}\tsupport={row['support']}"
        )


if __name__ == "__main__":
    main()
