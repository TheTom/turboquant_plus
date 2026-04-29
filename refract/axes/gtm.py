"""Axis A: GTM — Greedy Trajectory Match.

For each prompt in the dataset, greedy-decode N tokens from both the
reference (fp16-KV) and the candidate config. Compare token sequences.

Score mapping (per spec):
    GTM_score = 100 * full_match_rate

where full_match_rate is the fraction of prompts whose generated sequences
are token-identical between reference and candidate.

We additionally report:
    median_first_divergence_position
    mean_prefix_agreement_length
    per_prompt   (list of {id, ref, cand, first_div, prefix_len})

v0.1 simplification: we compare on the rendered text after stripping noise,
splitting on whitespace. v0.2 should call llama-tokenize on both completions
to get a true token-level diff that is robust to detokenisation collapse.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..runner import KVConfig, run_completion


@dataclass
class GTMResult:
    """Output of run_gtm()."""

    score: float                              # 0–100
    full_match_rate: float                    # 0–1
    median_first_divergence: Optional[int]    # token position; None if all match
    mean_prefix_agreement_length: float
    n_prompts: int
    n_tokens_each: int
    per_prompt: list[dict]


def _load_prompts(path: Path) -> list[dict]:
    """Load a JSONL prompts file. Each line: {id, category, prompt, ...}."""
    out = []
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            out.append(json.loads(ln))
    return out


def _tokenize_words(text: str) -> list[str]:
    """v0.1 token proxy: whitespace split.

    TODO(v0.2): pipe through `llama-tokenize -m MODEL` to compare on real
    BPE/SentencePiece tokens.
    """
    return text.split()


def _diff(ref: list[str], cand: list[str]) -> tuple[Optional[int], int]:
    """Return (first_divergence_position, prefix_agreement_length).

    first_divergence_position is None iff sequences are identical (treating
    cand as a candidate to match ref to length min(len)).
    """
    n = min(len(ref), len(cand))
    for i in range(n):
        if ref[i] != cand[i]:
            return i, i
    if len(ref) == len(cand):
        return None, n
    # one is prefix of the other — divergence is at the boundary
    return n, n


def run_gtm(
    model: Path,
    reference_kv: KVConfig,
    candidate_kv: KVConfig,
    prompts_path: Path,
    n_predict: int = 128,
    ctx: int = 512,
    n_gpu_layers: int = 99,
    seed: int = 42,
    progress: bool = True,
) -> GTMResult:
    """Run the GTM axis end to end.

    Greedy-decodes ``n_predict`` tokens from each prompt under both configs,
    then computes match statistics.
    """
    prompts = _load_prompts(prompts_path)
    if not prompts:
        raise ValueError(f"No prompts loaded from {prompts_path}")

    per_prompt: list[dict] = []
    matches = 0
    first_divs: list[int] = []
    prefix_lens: list[int] = []

    for i, p in enumerate(prompts):
        if progress:
            print(f"  [{i+1}/{len(prompts)}] {p['id']:<10} ({p.get('category','?')}) ...",
                  flush=True)

        ref_text, _ = run_completion(
            model=model, prompt=p["prompt"], kv=reference_kv,
            n_predict=n_predict, ctx=ctx, n_gpu_layers=n_gpu_layers, seed=seed,
        )
        cand_text, _ = run_completion(
            model=model, prompt=p["prompt"], kv=candidate_kv,
            n_predict=n_predict, ctx=ctx, n_gpu_layers=n_gpu_layers, seed=seed,
        )

        ref_toks = _tokenize_words(ref_text)
        cand_toks = _tokenize_words(cand_text)
        first_div, prefix_len = _diff(ref_toks, cand_toks)
        is_match = first_div is None

        if is_match:
            matches += 1
        else:
            first_divs.append(first_div)
        prefix_lens.append(prefix_len)

        per_prompt.append({
            "id": p["id"],
            "category": p.get("category"),
            "prompt": p["prompt"],
            "ref": ref_text,
            "cand": cand_text,
            "first_divergence": first_div,
            "prefix_agreement_length": prefix_len,
            "matched": is_match,
        })

    n = len(prompts)
    full_match_rate = matches / n
    median_first_div = statistics.median(first_divs) if first_divs else None
    mean_prefix = sum(prefix_lens) / n if n else 0.0
    score = 100.0 * full_match_rate

    return GTMResult(
        score=score,
        full_match_rate=full_match_rate,
        median_first_divergence=median_first_div,
        mean_prefix_agreement_length=mean_prefix,
        n_prompts=n,
        n_tokens_each=n_predict,
        per_prompt=per_prompt,
    )
