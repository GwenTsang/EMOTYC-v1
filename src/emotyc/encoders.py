from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from tokenizers import Tokenizer


class Encoder(Protocol):
    def encode(self, texts: list[str], batch_size: int) -> object:
        ...


@dataclass
class OnnxBackboneEncoder:
    session: object
    tokenizer: Tokenizer
    input_names: set[str]
    pad_id: int

    @classmethod
    def from_files(cls, backbone_path: str, tokenizer_path: str) -> "OnnxBackboneEncoder":
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        available_providers = ort.get_available_providers()
        preferred_providers = [
            provider
            for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in available_providers
        ]
        session = ort.InferenceSession(
            str(backbone_path),
            sess_options=options,
            providers=preferred_providers or None,
        )
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_truncation(max_length=512)
        pad_id = tokenizer.token_to_id("<pad>")
        if pad_id is None:
            pad_id = 1
        return cls(session, tokenizer, {item.name for item in session.get_inputs()}, int(pad_id))

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        ordered_items = self._sort_by_encoded_length(texts)
        outputs: list[tuple[list[int], np.ndarray]] = []
        batches = [
            ordered_items[start : start + batch_size]
            for start in range(0, len(ordered_items), batch_size)
        ]
        for items in reversed(batches):
            indices = [index for index, _ in items]
            batch = [text for _, text in items]
            if not batch:
                continue
            inputs = self._encode_batch(batch)
            backbone_output = self.session.run(None, inputs)[0]
            hidden = np.asarray(backbone_output, dtype=np.float32)
            if hidden.ndim != 3:
                raise ValueError(f"Backbone output must be 3D, got shape {hidden.shape}")
            outputs.append((indices, hidden[:, 0, :]))
        if not outputs:
            return np.empty((0, 0), dtype=np.float32)
        feature_dim = outputs[0][1].shape[1]
        restored = np.empty((len(texts), feature_dim), dtype=np.float32)
        for indices, features in outputs:
            restored[np.asarray(indices, dtype=np.int64)] = features
        return restored

    def _sort_by_encoded_length(self, texts: list[str]) -> list[tuple[int, str]]:
        lengths = [
            len(self.tokenizer.encode(text, add_special_tokens=False).ids)
            for text in texts
        ]
        return sorted(enumerate(texts), key=lambda item: lengths[item[0]])

    def _encode_batch(self, texts: list[str]) -> dict[str, np.ndarray]:
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        max_len = max((len(encoding.ids) for encoding in encodings), default=1)
        input_ids = np.full((len(encodings), max_len), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(encodings), max_len), dtype=np.int64)
        for row, encoding in enumerate(encodings):
            ids = encoding.ids or [self.pad_id]
            input_ids[row, : len(ids)] = ids
            attention_mask[row, : len(ids)] = 1

        inputs = {"input_ids": input_ids}
        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)
        return inputs
