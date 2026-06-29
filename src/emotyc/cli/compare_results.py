from __future__ import annotations

import argparse

from emotyc.compare import compare_metrics
from emotyc.io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Comparer deux fichiers metrics.json produits par evaluate.py.")
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--out", help="Chemin JSON de sortie optionnel.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    comparison = compare_metrics(args.a, args.b)
    print("Ecarts de metriques globales")
    for metric, values in comparison["global"].items():
        print(f"{metric}\tA={values['a']}\tB={values['b']}\tdelta={values['delta']}")

    print("")
    print("Ecarts par label")
    for label, metrics in comparison["per_label"].items():
        parts = [f"{name}={values['delta']}" for name, values in metrics.items()]
        print(f"{label}\t" + "\t".join(parts))

    print("")
    print("Labels seulement dans A: " + ", ".join(comparison["labels_only_a"]))
    print("Labels seulement dans B: " + ", ".join(comparison["labels_only_b"]))

    if args.out:
        write_json(args.out, comparison)


if __name__ == "__main__":
    main()
