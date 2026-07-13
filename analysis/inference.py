# -*- coding: utf-8 -*-
"""
Inference — EMOTYC Model Inference & Cached Prediction Loading
═══════════════════════════════════════════════════════════════
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from EMOTYC.common.common import DEFAULT_MODEL_NAME, get_predictor, predictor_from_paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_REPO = PROJECT_ROOT / "model_onnx" / "EMOTYC_ONNX_repo"
DEFAULT_MODEL_DIR = (
    DEFAULT_MODEL_REPO
    if DEFAULT_MODEL_REPO.exists()
    else PROJECT_ROOT / "model_onnx"
)
MODEL_DOWNLOAD_HINT = "Run `bash setup.sh` from the EMOTYC repository root."


# ═══════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _first_existing(paths):
    """Return the first existing path, or the first candidate for errors."""
    paths = [Path(path).expanduser() for path in paths]
    return next((path for path in paths if path.exists()), paths[0])


def _resolve_model_paths(model_path=None, tokenizer_path=None):
    """Resolve local ONNX artifacts, with CLI and environment overrides."""
    model_dir = Path(os.environ.get("EMOTYC_MODEL_DIR", DEFAULT_MODEL_DIR)).expanduser()

    if model_path is not None:
        onnx_path = Path(model_path).expanduser()
    elif os.environ.get("EMOTYC_ONNX_PATH"):
        onnx_path = Path(os.environ["EMOTYC_ONNX_PATH"]).expanduser()
    else:
        onnx_path = _first_existing([
            model_dir / "model.onnx",
            model_dir / "EMOTYC_2" / "model.onnx",
            PROJECT_ROOT / "model_onnx" / "model.onnx",
        ])

    if tokenizer_path is not None:
        tokenizer_path = Path(tokenizer_path).expanduser()
    elif os.environ.get("EMOTYC_TOKENIZER_PATH"):
        tokenizer_path = Path(os.environ["EMOTYC_TOKENIZER_PATH"]).expanduser()
    else:
        tokenizer_path = _first_existing([
            onnx_path.parent / "tokenizer.json",
            onnx_path.parent.parent / "tokenizer.json",
            model_dir / "tokenizer.json",
            PROJECT_ROOT / "model_onnx" / "tokenizer.json",
        ])
    return onnx_path, tokenizer_path


def _load_emotyc_model(device_name=None, model_path=None, tokenizer_path=None):
    """Charge le modèle EMOTYC ONNX et le tokenizer."""
    if device_name is not None:
        print("  Note: l'option device est ignorée par l'inférence ONNX canonique.")
    if model_path is None and tokenizer_path is None:
        predictor = get_predictor(model_dir=DEFAULT_MODEL_DIR, model_name=DEFAULT_MODEL_NAME)
        print(f"  ✓ Modèle EMOTYC ONNX chargé ({len(predictor.labels)} labels)")
        return predictor

    onnx_path, tokenizer_path = _resolve_model_paths(model_path, tokenizer_path)
    missing = [str(path) for path in (onnx_path, tokenizer_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing EMOTYC ONNX artifact(s): "
            + ", ".join(missing)
            + f". {MODEL_DOWNLOAD_HINT}"
        )
    predictor = predictor_from_paths(model_path=onnx_path, tokenizer_path=tokenizer_path)
    print(f"  ✓ Modèle EMOTYC ONNX chargé ({onnx_path})")
    print(f"  ✓ Tokenizer chargé depuis {tokenizer_path}")
    return predictor


def _predict_batch(predictor, texts, batch_size=16):
    """Inférence par batch → matrice (N, 19) de probas sigmoid."""
    return predictor.predict_texts(texts, batch_size=batch_size)


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def run_emotyc_inference(
    df,
    use_context=False,
    batch_size=16,
    device=None,
    model_path=None,
    tokenizer_path=None,
):
    """
    Exécute l'inférence EMOTYC sur tout le DataFrame.
    Ajoute les colonnes pred_* et proba_* pour chaque label.
    Envoie chaque phrase brute de la colonne TEXT au modèle.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'TEXT' column.
    use_context : bool
        Kept for CLI compatibility; ignored by this inference path.
    batch_size : int
        Inference batch size.
    device : str or None
        PyTorch device string.
    model_path : str or Path or None
        Explicit ONNX model path.
    tokenizer_path : str or Path or None
        Explicit tokenizer path. If omitted, resolved next to the model or
        from the parent ONNX repository.

    Returns
    -------
    pd.DataFrame
        Input df augmented with pred_* and proba_* columns.
    """
    predictor = _load_emotyc_model(device, model_path, tokenizer_path)

    # Préparer les phrases brutes, sans template before/current/after.
    input_texts = df["TEXT"].fillna("").astype(str).tolist()

    # Inférence
    print(f"\n  Inférence sur {len(df)} textes (batch_size={batch_size})…")
    all_probs_19 = _predict_batch(predictor, input_texts, batch_size)
    print(f"  ✓ Inférence terminée — shape: {all_probs_19.shape}")

    # Stocker probabilités et prédictions
    for gold_col, (emotyc_name, model_idx) in config.FULL_GOLD_TO_EMOTYC.items():
        proba_col = f"proba_{gold_col}"
        pred_col = f"pred_{gold_col}"
        df[proba_col] = all_probs_19[:, model_idx]

        df[pred_col] = (all_probs_19[:, model_idx] >= config.THRESHOLD).astype(int)

    return df


def load_cached_predictions(df, predictions_dir):
    """
    Charge un JSONL de prédictions produit par emotyc_predict.py.
    Joint les prédictions au DataFrame d'analyse.

    Parameters
    ----------
    df : pd.DataFrame
        Gold data in the same row order as the cached predictions.
    predictions_dir : str or Path
        Either a JSONL file path or a directory containing one
        emotyc_predictions.jsonl file.

    Returns
    -------
    pd.DataFrame
        Input df augmented with pred_* and proba_* columns.
    """
    predictions_path = Path(predictions_dir)
    if predictions_path.is_file():
        jsonl_path = predictions_path
    else:
        candidates = [predictions_path / "emotyc_predictions.jsonl"]
        candidates.extend(sorted(predictions_path.glob("**/emotyc_predictions.jsonl")))
        jsonl_path = next((path for path in candidates if path.exists()), None)
        if jsonl_path is None:
            raise FileNotFoundError(
                f"JSONL introuvable dans {predictions_path}. "
                "Attendu: emotyc_predictions.jsonl"
            )

    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if len(records) != len(df):
        raise ValueError(
            f"Mismatch predictions/gold: {len(records)} prédictions "
            f"vs {len(df)} lignes gold"
        )

    for global_idx, rec in zip(df.index.tolist(), records):
        # Émotions
        for emo in config.EMOTION_12:
            if emo in rec.get("preds", {}):
                df.at[global_idx, f"pred_{emo}"] = rec["preds"][emo]
                df.at[global_idx, f"proba_{emo}"] = rec["probas"].get(emo, np.nan)
        # Modes
        for mode in config.MODES_4:
            emotyc_mode = mode.replace("é", "e").replace("è", "e")
            if emotyc_mode in rec.get("preds_mode", {}):
                df.at[global_idx, f"pred_{mode}"] = rec["preds_mode"][emotyc_mode]
                df.at[global_idx, f"proba_{mode}"] = rec["probas_mode"].get(
                    emotyc_mode, np.nan
                )
        # Emo
        if "pred_emo" in rec:
            df.at[global_idx, "pred_Emo"] = rec["pred_emo"]
            df.at[global_idx, "proba_Emo"] = rec.get("proba_emo", np.nan)
        # Type
        for t in config.TYPES_2:
            if t in rec.get("preds_type", {}):
                df.at[global_idx, f"pred_{t}"] = rec["preds_type"][t]
                df.at[global_idx, f"proba_{t}"] = rec["probas_type"].get(t, np.nan)

    print(f"  ✓ Prédictions chargées depuis {jsonl_path}")
    return df


def run_or_load(df, args):
    """
    Convenience wrapper: runs inference or loads cached predictions
    based on CLI args.

    Parameters
    ----------
    df : pd.DataFrame
    args : argparse.Namespace
        Must have: skip_inference, predictions_dir, use_context,
        batch_size, device

    Returns
    -------
    pd.DataFrame
    """
    if args.skip_inference:
        return load_cached_predictions(df, args.predictions_dir)
    else:
        return run_emotyc_inference(
            df,
            use_context=args.use_context,
            batch_size=args.batch_size,
            device=args.device,
            model_path=getattr(args, "model_path", None),
            tokenizer_path=getattr(args, "tokenizer_path", None),
        )
