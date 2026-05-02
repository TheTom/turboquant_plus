"""Axis D (v0.2): PLAD — Perturbation-Locality-Aware Drift.

Probes brittleness: a quant whose answer flips when a single character of
the prompt changes. KLD@D + Trajectory + R-NIAH together cover short-
context distribution drift, trajectory-level token agreement, and long-
context retrieval, but none of them detect "the model worked fine on the
canned demo and got stupid as soon as a real user typed".

Protocol (v0.2.0)
-----------------

For each prompt P in the prompt set:

    1. Generate the *anchor* completion under both KV configs:
           a_ref  = generate(P, reference_kv)
           a_cand = generate(P, candidate_kv)

    2. For each perturbation type t in ``perturbations``:
           a. Construct P_t (perturbed prompt; see "Perturbation taxonomy").
           b. Generate:
                  p_ref  = generate(P_t, reference_kv)
                  p_cand = generate(P_t, candidate_kv)
           c. Drift = normalized token-edit distance between anchor and
              perturbed completions (token IDs from the model's vocab).
           d. ``excess_drift = max(0, cand_drift - ref_drift)``
           e. Score: ``plad_pp = 100 * exp(-alpha * excess_drift)``

Final score: arithmetic mean over all (prompt, perturbation) pairs.

Perturbation taxonomy
---------------------

  typo      Swap two adjacent characters in a randomly chosen non-stop
            word of length ≥ 4. Skipped if no eligible word exists.

  case      Lower-case the first character of every whitespace-token.
            Preserves all semantic content; punctuation untouched.

  punct     Remove the trailing question mark or period if present;
            otherwise append "?". Tests sensitivity to terminal
            punctuation, which sometimes acts as an EOS-like signal.

  paraphrase
            One-word lexical substitution from a small fixed synonym
            table. Conservative — must be transparently synonymous in
            context. Skipped if no eligible word exists.

Token edit distance is computed via :func:`tokenize_to_ids` and a
quadratic Levenshtein, so the metric is in the units the model
actually consumes. For typical n_predict ≤ 256 the cost is negligible
compared to the generation calls.

Implementation notes
--------------------

  - Anchor generation uses the same seed as perturbed generation so
    RNG variance doesn't inflate ``ref_drift`` and mask brittleness.
  - The perturbation set is deterministic given the prompt and a seed,
    so repeated runs reproduce.
  - Cost: O(n_prompts × (1 + n_perturbations) × 2) generations. With
    8 prompts and 4 perturbations that's 80 calls; at ~5 sec/call ~7
    minutes per matrix cell.
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..runner import KVConfig, run_completion, tokenize_to_ids

DEFAULT_PERTURBATIONS = ("typo", "case", "punct", "paraphrase")
DEFAULT_ALPHA = 5.0

# Minimal closed-vocabulary synonym set for the paraphrase perturbation.
# Conservative — every entry is a transparent synonym in default English
# usage. Add carefully; an entry that's only sometimes synonymous (e.g.
# capital → seat) shifts the test from "brittleness" to "semantic
# fragility" and inflates excess_drift everywhere.
_SYNONYMS: dict[str, str] = {
    "big":     "large",
    "large":   "big",
    "small":   "tiny",
    "tiny":    "small",
    "fast":    "quick",
    "quick":   "fast",
    "begin":   "start",
    "start":   "begin",
    "happy":   "glad",
    "sad":     "unhappy",
    "smart":   "clever",
    "clever":  "smart",
    "show":    "display",
    "display": "show",
    "build":   "construct",
    "create":  "make",
    "make":    "create",
    "find":    "locate",
    "locate":  "find",
}

# Stopwords skipped by the typo and paraphrase perturbations so we don't
# insert obviously-broken function words like "a"/"the" → "ahe"/"teh".
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "of", "for", "with", "by", "as", "and", "or",
    "but", "not", "no", "do", "does", "did", "i", "you", "he", "she", "it",
    "we", "they", "this", "that", "these", "those", "have", "has", "had",
}


@dataclass
class PLADPerPrompt:
    prompt_id: str
    perturbation: str
    perturbed_prompt: str
    ref_drift: float
    cand_drift: float
    excess_drift: float
    plad_pp: float


@dataclass
class PLADResult:
    score: float
    per_perturbation_score: dict[str, float]
    per_prompt: list[PLADPerPrompt]
    n_prompts: int
    n_perturbations: int
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Perturbation generators
# ---------------------------------------------------------------------------


_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def _eligible_words(prompt: str) -> list[tuple[int, int, str]]:
    """Return (start, end, word) triples for non-stopword whitespace-tokens
    of length ≥ 2 (typo wants ≥ 4; paraphrase wants ≥ 3)."""
    out = []
    for m in _WORD_RE.finditer(prompt):
        w = m.group(0)
        if w.lower() in _STOPWORDS:
            continue
        out.append((m.start(), m.end(), w))
    return out


def _apply_typo(prompt: str, rng: random.Random) -> Optional[str]:
    eligible = [(s, e, w) for (s, e, w) in _eligible_words(prompt) if len(w) >= 4]
    if not eligible:
        return None
    s, e, w = rng.choice(eligible)
    # Swap two adjacent characters at a random interior position
    i = rng.randrange(0, len(w) - 1)
    swapped = w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if swapped == w:  # palindromic pair, e.g. "ee"; skip
        return None
    return prompt[:s] + swapped + prompt[e:]


def _apply_case(prompt: str) -> Optional[str]:
    out = []
    last_end = 0
    changed = False
    for m in _WORD_RE.finditer(prompt):
        out.append(prompt[last_end:m.start()])
        w = m.group(0)
        if w[0].isupper():
            out.append(w[0].lower() + w[1:])
            changed = True
        else:
            out.append(w)
        last_end = m.end()
    out.append(prompt[last_end:])
    if not changed:
        return None
    return "".join(out)


def _apply_punct(prompt: str) -> Optional[str]:
    p = prompt.rstrip()
    if p.endswith("?"):
        return p[:-1] + (prompt[len(p):])
    if p.endswith("."):
        return p[:-1] + (prompt[len(p):])
    return p + "?" + prompt[len(p):]


def _apply_paraphrase(prompt: str, rng: random.Random) -> Optional[str]:
    eligible = [
        (s, e, w) for (s, e, w) in _eligible_words(prompt)
        if len(w) >= 3 and w.lower() in _SYNONYMS
    ]
    if not eligible:
        return None
    s, e, w = rng.choice(eligible)
    sub = _SYNONYMS[w.lower()]
    # Preserve the surface case of the original (only first-letter capitalization).
    if w[0].isupper():
        sub = sub[0].upper() + sub[1:]
    return prompt[:s] + sub + prompt[e:]


_PERTURBATION_FUNCS = {
    "typo":       _apply_typo,
    "case":       lambda p, _rng: _apply_case(p),
    "punct":      lambda p, _rng: _apply_punct(p),
    "paraphrase": _apply_paraphrase,
}


# ---------------------------------------------------------------------------
# Token edit distance
# ---------------------------------------------------------------------------


def _levenshtein(a: list[int], b: list[int]) -> int:
    """Quadratic Levenshtein on integer token-ID sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, 1):
            cost = 0 if ai == bj else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def _normalized_drift(model: Path, anchor: str, perturbed: str) -> float:
    """Drift = token-edit-distance(anchor, perturbed) / len(anchor).

    Returns 0.0 when both are empty; 1.0 when anchor is empty but
    perturbed isn't (max drift). Capped at 1.0 to keep the metric in
    [0, 1] regardless of length differences.
    """
    a_tok = tokenize_to_ids(model, anchor) if anchor else []
    p_tok = tokenize_to_ids(model, perturbed) if perturbed else []
    if not a_tok and not p_tok:
        return 0.0
    if not a_tok:
        return 1.0
    d = _levenshtein(a_tok, p_tok)
    return min(1.0, d / len(a_tok))


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def _load_prompts(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            out.append(json.loads(ln))
    return out


def run_plad(
    model: Path,
    prompts_path: Path,
    reference_kv: KVConfig,
    candidate_kv: KVConfig,
    n_predict: int = 64,
    ctx: int = 512,
    n_gpu_layers: int = 99,
    seed: int = 42,
    perturbations: tuple[str, ...] = DEFAULT_PERTURBATIONS,
    alpha: float = DEFAULT_ALPHA,
    progress: bool = True,
) -> PLADResult:
    """Run PLAD end to end.

    Returns a :class:`PLADResult` with per-(prompt, perturbation) drift
    diagnostics and per-perturbation aggregate scores in addition to the
    overall composite. (prompt, perturbation) cells where the perturbation
    couldn't apply (e.g. ``typo`` on a prompt with no ≥4-char words) are
    skipped and reported in ``notes``.
    """
    prompts = _load_prompts(prompts_path)
    if not prompts:
        raise ValueError(f"No prompts loaded from {prompts_path}")
    unknown = [p for p in perturbations if p not in _PERTURBATION_FUNCS]
    if unknown:
        raise ValueError(f"Unknown perturbations: {unknown!r}; "
                         f"valid: {tuple(_PERTURBATION_FUNCS)}")

    per_prompt_records: list[PLADPerPrompt] = []
    per_pert_scores: dict[str, list[float]] = {p: [] for p in perturbations}
    skipped_count = 0
    notes: list[str] = []

    # Build the full (prompt, perturbation_or_None) plan first so we can
    # batch all ref-side completions before any cand-side completions.
    # Memory-pressured backends (vLLM on hybrid Qwen3.6) can only hold one
    # LLM at a time, so interleaving ref/cand per prompt forces N model
    # reloads — this batching keeps it to two.
    Cell = tuple[int, str | None, str]  # (prompt_idx, perturbation, prompt_text)
    cells: list[Cell] = []
    for i, p in enumerate(prompts):
        cells.append((i, None, p["prompt"]))
        rng = random.Random(seed + i)
        for pert in perturbations:
            fn = _PERTURBATION_FUNCS[pert]
            perturbed = fn(p["prompt"], rng)
            if perturbed is None or perturbed == p["prompt"]:
                skipped_count += 1
                continue
            cells.append((i, pert, perturbed))

    ref_text: dict[tuple[int, str | None], str] = {}
    cand_text: dict[tuple[int, str | None], str] = {}

    for idx, (prompt_idx, pert_name, prompt_text) in enumerate(cells):
        if progress:
            label = "anchor" if pert_name is None else f"pert={pert_name}"
            print(f"  ref [{idx + 1}/{len(cells)}] p{prompt_idx} {label} ...", flush=True)
        text, _ = run_completion(
            model=model, prompt=prompt_text, kv=reference_kv,
            n_predict=n_predict, ctx=ctx, n_gpu_layers=n_gpu_layers, seed=seed,
        )
        ref_text[(prompt_idx, pert_name)] = text
    for idx, (prompt_idx, pert_name, prompt_text) in enumerate(cells):
        if progress:
            label = "anchor" if pert_name is None else f"pert={pert_name}"
            print(f"  cand [{idx + 1}/{len(cells)}] p{prompt_idx} {label} ...", flush=True)
        text, _ = run_completion(
            model=model, prompt=prompt_text, kv=candidate_kv,
            n_predict=n_predict, ctx=ctx, n_gpu_layers=n_gpu_layers, seed=seed,
        )
        cand_text[(prompt_idx, pert_name)] = text

    for i, p in enumerate(prompts):
        a_ref = ref_text.get((i, None), "")
        a_cand = cand_text.get((i, None), "")
        for pert in perturbations:
            key = (i, pert)
            if key not in ref_text:
                continue
            p_ref = ref_text[key]
            p_cand = cand_text[key]
            ref_drift = _normalized_drift(model, a_ref, p_ref)
            cand_drift = _normalized_drift(model, a_cand, p_cand)
            excess = max(0.0, cand_drift - ref_drift)
            plad_pp = 100.0 * math.exp(-alpha * excess)
            per_prompt_records.append(PLADPerPrompt(
                prompt_id=str(p["id"]),
                perturbation=pert,
                perturbed_prompt=cells[next(
                    j for j, c in enumerate(cells) if c[0] == i and c[1] == pert
                )][2],
                ref_drift=ref_drift,
                cand_drift=cand_drift,
                excess_drift=excess,
                plad_pp=plad_pp,
            ))
            per_pert_scores[pert].append(plad_pp)

    if not per_prompt_records:
        raise RuntimeError(
            "PLAD ran zero (prompt, perturbation) cells. Either every "
            "perturbation skipped or no prompts were loaded."
        )

    if skipped_count:
        notes.append(
            f"{skipped_count} (prompt, perturbation) pairs were skipped "
            f"(perturbation could not apply, e.g. no ≥4-char word for typo)."
        )

    overall = statistics.mean(r.plad_pp for r in per_prompt_records)
    per_pert_summary = {
        pert: (statistics.mean(scores) if scores else float("nan"))
        for pert, scores in per_pert_scores.items()
    }
    return PLADResult(
        score=overall,
        per_perturbation_score=per_pert_summary,
        per_prompt=per_prompt_records,
        n_prompts=len(prompts),
        n_perturbations=len(perturbations),
        notes=notes,
    )
