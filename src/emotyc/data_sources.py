from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

from emotyc.registry import DATASET_ALIASES, resolve_dataset_entry


def resolve_dataset(alias: str) -> Path:
    entry = resolve_dataset_entry(alias)
    return Path(hf_hub_download(repo_id=entry["repo_id"], filename=entry["filename"], repo_type="dataset"))


def resolve_data(value: str) -> Path:
    if value in DATASET_ALIASES:
        return resolve_dataset(value)
    path = Path(value).expanduser()
    if path.suffix.lower() != ".xlsx":
        raise ValueError(f"Data path must be an .xlsx file or a known dataset alias: {value}")
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    return path
