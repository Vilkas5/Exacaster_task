"""Local text embeddings — no API key, no per-call cost, no network at
runtime. Runs a quantized ONNX export of all-MiniLM-L6-v2 (~23MB, committed
under model/) via onnxruntime instead of the full sentence-transformers +
PyTorch stack, which would be hundreds of MB and risk exceeding Streamlit
Community Cloud's free-tier memory/build limits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

_MODEL_DIR = Path(__file__).parent / "model"

_tokenizer: Tokenizer | None = None
_session: ort.InferenceSession | None = None


def _load() -> tuple[Tokenizer, ort.InferenceSession]:
    global _tokenizer, _session
    if _tokenizer is None:
        _tokenizer = Tokenizer.from_file(str(_MODEL_DIR / "tokenizer.json"))
        _tokenizer.enable_padding()
        _tokenizer.enable_truncation(max_length=256)
    if _session is None:
        _session = ort.InferenceSession(
            str(_MODEL_DIR / "embedding_model.onnx"),
            providers=["CPUExecutionProvider"],
        )
    return _tokenizer, _session


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalized sentence embeddings, shape (len(texts), 384)."""
    tokenizer, session = _load()
    encodings = tokenizer.encode_batch(texts)

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)

    (last_hidden_state,) = session.run(
        ["last_hidden_state"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )

    # Mean-pool token embeddings, ignoring padding positions.
    mask = attention_mask[:, :, None].astype(np.float32)
    summed = (last_hidden_state * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    pooled = summed / counts

    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.clip(norms, a_min=1e-9, a_max=None)
