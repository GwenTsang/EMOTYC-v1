from __future__ import annotations

import argparse
from pathlib import Path

from emotyc.artifacts import resolve_model_bundle
from emotyc.data_sources import resolve_data, resolve_dataset
from emotyc.datasets import load_xlsx
from emotyc.formatting import apply_template
from emotyc.io import export_predictions_xlsx, write_json
from emotyc.metrics import compute_metrics
from emotyc.predictors import Predictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluer les labels EMOTYC sur un fichier XLSX annote.")
    parser.add_argument("--model", required=True, help="Alias modele connu ou bundle modele v1 local.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="Alias dataset connu.")
    source.add_argument("--data", help="Fichier XLSX local ou alias dataset connu.")
    parser.add_argument("--template", choices=("raw", "bca"), default="raw")
    parser.add_argument(
        "--use-context",
        action="store_true",
        help="Utiliser les lignes voisines i-1 et i+1 dans le template bca.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Seuil global de prediction.")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille de batch.")
    parser.add_argument("--save-config", action="store_true", help="Sauver config.json.")
    parser.add_argument("--save-metrics", action="store_true", help="Sauver metrics.json.")
    parser.add_argument("--save-predictions", action="store_true", help="Sauver predictions.xlsx.")
    parser.add_argument("--out-dir", default=".", help="Dossier de sortie.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_path = resolve_dataset(args.dataset) if args.dataset else resolve_data(args.data)
    bundle = resolve_model_bundle(args.model)
    predictor = Predictor.from_bundle(bundle)
    dataset = load_xlsx(data_path, model_labels=predictor.labels, require_gold=True)
    texts = apply_template(dataset.texts, args.template, use_context=args.use_context)
    result = predictor.predict(texts, batch_size=args.batch_size, threshold=args.threshold)

    indices = [predictor.labels.index(label) for label in dataset.labels_evaluated]
    pred = result.predictions[:, indices]
    proba = result.probabilities[:, indices]
    if dataset.gold is None:
        raise ValueError("Le fichier XLSX ne contient aucune colonne label exploitable.")
    global_metrics, per_label = compute_metrics(dataset.gold, pred, dataset.labels_evaluated)

    print("Metriques globales")
    print(format_global_metrics_table(global_metrics))
    print("")
    print("Metriques par label")
    print(format_per_label_metrics_table(per_label))
    if dataset.labels_missing:
        print("")
        print("Labels absents: " + ", ".join(dataset.labels_missing))

    if args.save_config or args.save_metrics or args.save_predictions:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.save_config:
            config = {
                "model": args.model,
                "dataset": args.dataset,
                "data": args.data,
                "template": args.template,
                "use_context": args.use_context,
                "threshold": args.threshold,
                "batch_size": args.batch_size,
            }
            write_json(out_dir / "config.json", config)
        if args.save_metrics:
            metrics = {
                "global_metrics": global_metrics,
                "per_label": per_label,
                "labels_evaluated": dataset.labels_evaluated,
                "n_samples": len(dataset.texts),
                "threshold": args.threshold,
            }
            write_json(out_dir / "metrics.json", metrics)
        if args.save_predictions:
            export_predictions_xlsx(
                out_dir / "predictions.xlsx",
                texts=dataset.texts,
                labels=dataset.labels_evaluated,
                predictions=pred,
                probabilities=proba,
                gold=dataset.gold,
            )


def format_global_metrics_table(metrics: dict[str, object]) -> str:
    """Format global evaluation metrics as a terminal-friendly Markdown table."""
    return "\n".join(
        (
            "| micro-F1 | macro-F1 | exact match |",
            "|---:|---:|---:|",
            "| {micro_f1:.4f} | {macro_f1:.4f} | {exact_match:.4f} |".format(**metrics),
        )
    )


def format_per_label_metrics_table(rows: list[dict[str, object]]) -> str:
    """Format precision, recall, F1 and support for each evaluated label."""
    lines = [
        "| Label | Précision | Rappel | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {support} |".format(
                **row
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
