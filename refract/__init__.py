"""REFRACT v0.1 — REFerence-anchored Robust Acid-test for Compressed Transformers.

A benchmaxx-resistant alternative to corpus PPL for evaluating KV-cache
quantization quality. Replaces "lower PPL = better" with a composite of:

  - Axis A (GTM): Greedy Trajectory Match — do quantized + reference produce
    the same generated tokens?
  - Axis B (KLD@D): KL Divergence vs the fp16-KV reference distribution
    (v0.1 uses corpus KLD via llama-perplexity as a proxy for trajectory KLD).

See:
  - refract/README.md for usage.
  - docs/papers/attn-rotation-and-ppl-artifact.md for the motivating paper.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
