"""REFRACT backend abstraction.

Each backend (llama.cpp, MLX, vLLM) implements the :class:`Backend` ABC.
Axes call into the backend rather than into a hardcoded subprocess
wrapper, so the same scoring framework works on any inference engine
that can give us:

  - text-in / text-out completion under a given KV config
  - per-token model-token IDs at decode time (for trajectory)
  - per-token KL divergence vs a reference (for KLD@D)
  - chat-template application (so instruct models engage Q&A mode)
  - tokenization to integer ID list (for PLAD edit distance)

Backend selection
-----------------

`get_backend(name)`:
    Explicit selection by name ('llamacpp' | 'mlx' | 'vllm').

`auto_backend(model)`:
    Pick by inspecting the path. ``.gguf`` → llama.cpp;
    a directory with ``model.safetensors.index.json`` and
    ``mlx_*`` shape → MLX; anything else → vLLM.

Override via env var ``REFRACT_BACKEND``.

Status (v0.3.2):

  - llamacpp: production (primary dev target on macOS Apple Silicon
              + Linux Ubuntu via the patched binary)
  - mlx:      production (Apple Silicon native; mlx-lm 0.31+)
  - vllm:     production (HF safetensors on CUDA / ROCm; cached
              in-process LLM with evict-on-key-change for memory-pressured
              hybrid models; verified on AMD MI300X, ROCm 7.2)
  - sglang:   production (HTTP client; SGLang server runs separately,
              typically Docker; verified on AMD MI300X, ROCm 7.2)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .base import Backend, BackendCapabilityError


def get_backend(name: str) -> Backend:
    """Return a backend instance by name. Raises ValueError for unknown."""
    name = name.lower()
    if name == "llamacpp":
        from .llamacpp import LlamaCppBackend
        return LlamaCppBackend()
    if name == "mlx":
        from .mlx import MLXBackend
        return MLXBackend()
    if name == "vllm":
        from .vllm import VLLMBackend
        return VLLMBackend()
    if name == "sglang":
        from .sglang import SGLangBackend
        return SGLangBackend()
    raise ValueError(
        f"Unknown backend {name!r}. Valid: 'llamacpp', 'mlx', 'vllm', 'sglang'."
    )


def auto_backend(model: Path) -> Backend:
    """Pick the backend by inspecting the model path + REFRACT_BACKEND env.

    Resolution order:
      1. ``REFRACT_BACKEND`` env var (explicit override)
      2. Path suffix: ``.gguf`` → llama.cpp
      3. Directory containing ``model.safetensors`` and a recognisable
         MLX-shaped tensor index → mlx
      4. Anything else → vllm (which loads via Hugging Face IDs or local dirs)
    """
    env = os.environ.get("REFRACT_BACKEND")
    if env:
        return get_backend(env)
    if model.suffix == ".gguf":
        return get_backend("llamacpp")
    if model.is_dir():
        # Heuristic: MLX models ship with a tokenizer.json + model.safetensors
        # plus an mlx-style config; vllm/HF models ship the same files but
        # without the mlx indicator. Default to mlx for now; users can
        # override via REFRACT_BACKEND=vllm.
        if (model / "config.json").exists():
            return get_backend("mlx")
    return get_backend("vllm")


__all__ = [
    "Backend",
    "BackendCapabilityError",
    "get_backend",
    "auto_backend",
]
