MODEL_ALIASES = {
    "emotyc_1": "GwendalTsang/EMOTYC_1",
    "emotyc_2": "GwendalTsang/EMOTYC_2",
}

DATASET_ALIASES = {
    "CyberAgg": {
        "repo_id": "GwendalTsang/CyberAggAdo",
        "filename": "CyberAdoAgg_gold_global_total_latest.xlsx",
    },
    "ttk": {
        "repo_id": "GwendalTsang/TTK",
        "filename": "emotexttokids_gold_flat.xlsx",
    },
}


def model_aliases() -> list[str]:
    return sorted(MODEL_ALIASES)


def dataset_aliases() -> list[str]:
    return sorted(DATASET_ALIASES)


def resolve_model_repo(alias: str) -> str:
    try:
        return MODEL_ALIASES[alias]
    except KeyError as exc:
        available = ", ".join(model_aliases())
        raise ValueError(f"Unknown model alias '{alias}'. Available aliases: {available}") from exc


def resolve_dataset_entry(alias: str) -> dict[str, str]:
    try:
        return DATASET_ALIASES[alias]
    except KeyError as exc:
        available = ", ".join(dataset_aliases())
        raise ValueError(f"Unknown dataset alias. Available aliases: {available}") from exc
