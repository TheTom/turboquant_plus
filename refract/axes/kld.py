"""Axis B: KLD@D — KL Divergence vs reference distribution.

v0.1 approximation: corpus KLD via ``llama-perplexity --kl-divergence``.
This is a usable proxy for trajectory-KLD because:

  - the noise floor on this codepath is bit-exact zero on Metal builds
    (paper §4.5), so any non-trivial KLD is signal not noise.
  - on the reference cases in the paper (gemma-4 26B-A4B, Qwen2.5-7B,
    gemma-4 E2B) corpus-KLD is the metric that ranks configurations
    correctly while corpus-PPL inverts.

Score mapping (per spec):
    KLD_score = 100 * exp(-mean_kld)

so 0 nats → 100, 0.7 nats → 50, 1.7 nats (paper headline) → ~18.

TODO(v0.2): replace with true trajectory-KLD by capturing per-step logits
during generation in the GTM pass — same forward, two oracles.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..runner import KVConfig, run_perplexity_kld, run_perplexity_kld_base


@dataclass
class KLDResult:
    score: float                 # 0–100
    mean_kld: float              # nats
    ppl: Optional[float]
    rms_dp_pct: Optional[float]
    same_topp_pct: Optional[float]
    base_path: str
    chunks: int
    ctx: int
    is_self_reference: bool      # True when candidate == reference; should give 0


def _kld_to_score(kld: float) -> float:
    """Map mean KLD (nats) → 0–100 with score = 100 * exp(-kld)."""
    if kld < 0:
        # llama-perplexity should never report negative; clamp defensively
        kld = 0.0
    return 100.0 * math.exp(-kld)


def run_kld(
    model: Path,
    corpus: Path,
    reference_kv: KVConfig,
    candidate_kv: KVConfig,
    chunks: int = 32,
    ctx: int = 512,
    n_gpu_layers: int = 99,
    base_path: Optional[Path] = None,
    progress: bool = True,
) -> KLDResult:
    """Run the KLD axis end to end.

    1. If base_path is None, build an fp16-KV reference base in a temp file
       (will be deleted on success).
    2. Score candidate against the base.
    3. Map mean KLD → 0–100 score.
    """
    cleanup_base = False
    if base_path is None:
        # tempfile.mkstemp gives us a path we own; llama-perplexity will
        # write its own format, we just need a writable location.
        fd, path = tempfile.mkstemp(prefix="refract-kldbase-", suffix=".bin")
        os.close(fd)
        os.unlink(path)  # llama-perplexity creates it itself
        base_path = Path(path)
        cleanup_base = True

    is_self_ref = (reference_kv.label() == candidate_kv.label())

    try:
        if progress:
            print(f"  [1/2] Building fp16-KV reference base: {reference_kv.label()}",
                  flush=True)
        run_perplexity_kld_base(
            model=model, corpus=corpus, kv=reference_kv,
            base_path=base_path, chunks=chunks, ctx=ctx,
            n_gpu_layers=n_gpu_layers,
        )

        if progress:
            print(f"  [2/2] Scoring candidate: {candidate_kv.label()}", flush=True)
        scored = run_perplexity_kld(
            model=model, corpus=corpus, kv=candidate_kv,
            base_path=base_path, chunks=chunks, ctx=ctx,
            n_gpu_layers=n_gpu_layers,
        )

        result = KLDResult(
            score=_kld_to_score(scored["mean_kld"]),
            mean_kld=scored["mean_kld"],
            ppl=scored.get("ppl"),
            rms_dp_pct=scored.get("rms_dp_pct"),
            same_topp_pct=scored.get("same_topp_pct"),
            base_path=str(base_path),
            chunks=chunks,
            ctx=ctx,
            is_self_reference=is_self_ref,
        )
    finally:
        if cleanup_base and base_path.exists():
            try:
                base_path.unlink()
            except OSError:
                pass

    return result
