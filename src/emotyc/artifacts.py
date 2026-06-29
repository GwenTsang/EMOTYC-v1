from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from emotyc.registry import resolve_model_repo

REQUIRED_MODEL_FILES = (
    "backbone.onnx",
    "head.npz",
    "head.json",
    "tokenizer.json",
    "model_config.json",
)


@dataclass(frozen=True)
class ModelBundle:
    root: Path
    backbone: Path
    head_weights: Path
    head_config: Path
    tokenizer: Path
    model_config: Path


def resolve_model_bundle(model: str) -> ModelBundle:
    path = Path(model).expanduser()
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Local model path is not a directory: {path}")
        return bundle_from_directory(path)

    repo_id = resolve_model_repo(model)
    paths = {
        filename: Path(hf_hub_download(repo_id=repo_id, filename=filename))
        for filename in REQUIRED_MODEL_FILES
    }
    return ModelBundle(
        root=paths["backbone.onnx"].parent,
        backbone=paths["backbone.onnx"],
        head_weights=paths["head.npz"],
        head_config=paths["head.json"],
        tokenizer=paths["tokenizer.json"],
        model_config=paths["model_config.json"],
    )


def bundle_from_directory(path: str | Path) -> ModelBundle:
    path = Path(path)
    missing = [filename for filename in REQUIRED_MODEL_FILES if not (path / filename).is_file()]
    if missing:
        raise ValueError(
            f"Incomplete model bundle at {path}. Missing files: {', '.join(missing)}"
        )
    return ModelBundle(
        root=path,
        backbone=path / "backbone.onnx",
        head_weights=path / "head.npz",
        head_config=path / "head.json",
        tokenizer=path / "tokenizer.json",
        model_config=path / "model_config.json",
    )
