from __future__ import annotations

import argparse

from emotyc.artifacts import resolve_model_bundle
from emotyc.data_sources import resolve_data
from emotyc.datasets import load_xlsx
from emotyc.io import export_predictions_xlsx
from emotyc.predictors import Predictor
from emotyc.formatting import apply_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predire les labels EMOTYC pour un fichier XLSX.")
    parser.add_argument("--model", required=True, help="Alias modele connu ou bundle modele v1 local.")
    parser.add_argument("--data", required=True, help="Fichier XLSX local.")
    parser.add_argument("--template", choices=("raw", "bca"), default="raw")
    parser.add_argument(
        "--use-context",
        action="store_true",
        help="Utiliser les lignes voisines i-1 et i+1 dans le template bca.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Seuil global de prediction.")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille de batch.")
    parser.add_argument("--out", help="Chemin XLSX de sortie optionnel.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    bundle = resolve_model_bundle(args.model)
    predictor = Predictor.from_bundle(bundle)
    data_path = resolve_data(args.data)
    dataset = load_xlsx(data_path)
    texts = apply_template(dataset.texts, args.template, use_context=args.use_context)
    result = predictor.predict(texts, batch_size=args.batch_size, threshold=args.threshold)

    for index, row in enumerate(result.predictions):
        positives = [label for label, value in zip(result.labels, row, strict=True) if value == 1]
        print(f"{index + 1}\t{', '.join(positives)}")

    if args.out:
        export_predictions_xlsx(
            args.out,
            texts=dataset.texts,
            labels=result.labels,
            predictions=result.predictions,
            probabilities=result.probabilities,
        )


if __name__ == "__main__":
    main()
