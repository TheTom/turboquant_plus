"""Validation: REFRACT must correctly flag the gemma-4 26B-A4B q8/turbo4 OFF
cell from the paper as DEGRADED or worse.

This is the cell from §4.3 + §4.8 of
docs/papers/attn-rotation-and-ppl-artifact.md:

    config        : ctk=q8_0,ctv=turbo4,attn_rot_v=0,attn_rot_k=0
    reference     : ctk=f16,ctv=f16
    paper KLD     : 1.738 ± 0.036 nats        → KLD_score ≈ 100*exp(-1.738) ≈ 17.6
    paper §4.8 GTM (3 prompts, 40 tok): 1/3 fully matched (capital of France)
                                        2/3 token-divergent within 4 tokens
                                        → GTM_score ≈ 33 (3 prompts) but with
                                          our 30-prompt set the prediction is
                                          looser; we just check the band.

Expected REFRACT band: DEGRADED (60-79) or FAIL (<60). EXCELLENT/PASS would
be a regression — corpus PPL on this cell is -42% (apparent "win"); REFRACT
must NOT mistake that for a pass.

This test is integration-only: it spawns llama-cli + llama-perplexity many
times against a 26B-A4B model and takes 30+ minutes. Run with:

    pytest -m integration refract/tests/test_validation.py

or directly:

    python3 -m refract.cli score \\
        --model ~/local_llms/models/gemma-4-26B-A4B-Q8_0.gguf \\
        --reference 'ctk=f16,ctv=f16' \\
        --candidate 'ctk=q8_0,ctv=turbo4,attn_rot_v=0,attn_rot_k=0' \\
        --prompts refract/prompts/v0.1.jsonl \\
        --corpus ~/local_llms/llama.cpp/wikitext-2-raw/wiki.test.raw \\
        --chunks 32 --measure-floor
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from refract.axes.gtm import run_gtm
from refract.axes.kld import run_kld
from refract.runner import KVConfig
from refract.score import composite_score


# Default paths; override via env if your model lives elsewhere.
GEMMA_MODEL = Path(os.environ.get(
    "REFRACT_GEMMA_MODEL",
    os.path.expanduser("~/local_llms/models/gemma-4-26B-A4B-Q8_0.gguf"),
))
WIKITEXT = Path(os.environ.get(
    "REFRACT_CORPUS",
    os.path.expanduser("~/local_llms/llama.cpp/wikitext-2-raw/wiki.test.raw"),
))
PROMPTS = Path(__file__).resolve().parent.parent / "prompts" / "v0.1.jsonl"


@pytest.mark.integration
@pytest.mark.skipif(
    not GEMMA_MODEL.exists(),
    reason=f"gemma-4 26B-A4B model not found at {GEMMA_MODEL}; "
           "set REFRACT_GEMMA_MODEL to override.",
)
@pytest.mark.skipif(
    not WIKITEXT.exists(),
    reason=f"wikitext-2 raw not found at {WIKITEXT}; "
           "set REFRACT_CORPUS to override.",
)
def test_gemma_q8_turbo4_off_is_degraded():
    """Reproduce the paper §4.3 + §4.8 gemma-4 26B-A4B cell.

    Expected: REFRACT band in {DEGRADED, FAIL}, NOT in {PASS, EXCELLENT}.
    PPL on this cell would say "win" (-42%); REFRACT must say "audit" or "fail".
    """
    ref_kv = KVConfig.parse("ctk=f16,ctv=f16")
    cand_kv = KVConfig.parse("ctk=q8_0,ctv=turbo4,attn_rot_v=0,attn_rot_k=0")

    gtm = run_gtm(
        model=GEMMA_MODEL,
        reference_kv=ref_kv,
        candidate_kv=cand_kv,
        prompts_path=PROMPTS,
        n_predict=128,
        ctx=512,
        n_gpu_layers=99,
        seed=42,
        progress=True,
    )
    kld = run_kld(
        model=GEMMA_MODEL,
        corpus=WIKITEXT,
        reference_kv=ref_kv,
        candidate_kv=cand_kv,
        chunks=32,
        ctx=512,
        n_gpu_layers=99,
        progress=True,
    )

    composite = composite_score(gtm.score, kld.score)

    print(f"\nGTM score: {gtm.score:.2f}")
    print(f"KLD score: {kld.score:.2f}  (mean KLD = {kld.mean_kld:.4f})")
    print(f"REFRACT  : {composite.composite:.2f}  band={composite.band}")

    # Hard floor: paper KLD = 1.7 nats → KLD_score < 25 → composite < 50 unless
    # GTM is very good. Paper §4.8 shows 2/3 prompts diverged within 4 tokens
    # at 40 generated tokens; our 30-prompt 128-token decode is more demanding,
    # so GTM will be ≤ §4.8 rate at best.
    assert composite.band in {"DEGRADED", "FAIL"}, (
        f"REFRACT mistakenly flagged the gemma-4 q8/turbo4 OFF cell as "
        f"{composite.band} ({composite.composite:.2f}). PPL says this config "
        f"'wins' by -42%; REFRACT must say it is degraded. "
        f"Diagnostics: gtm={gtm.score:.2f}, kld={kld.score:.2f} "
        f"(mean KLD={kld.mean_kld:.4f} nats)."
    )

    # Sanity: the KLD axis on its own should land in DEGRADED/FAIL too,
    # because the paper's measured KLD is ~1.7 nats.
    assert kld.score < 60.0, (
        f"KLD score {kld.score:.2f} unexpectedly high; paper measured "
        f"~1.7 nats KLD which should map to KLD_score≈17.6."
    )
