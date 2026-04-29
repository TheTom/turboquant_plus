# REFRACT v0.1

**REF**erence-anchored **R**obust **A**cid-test for **C**ompressed **T**ransformers.

A benchmaxx-resistant alternative to corpus PPL for evaluating KV-cache
quantization quality. Replaces "lower PPL = better" — a metric the paper
[`docs/papers/attn-rotation-and-ppl-artifact.md`](../../docs/papers/attn-rotation-and-ppl-artifact.md)
shows can invert sign on instruct-tuned models — with a composite of two
fp16-anchored axes that ranks configurations by *distance from the fp16-KV
reference*, not by absolute corpus likelihood.

## What v0.1 ships

Two of the four planned axes:

| Axis | What it measures | Scoring |
|------|------------------|---------|
| **A — GTM** (Greedy Trajectory Match) | For K diverse prompts, do the reference and candidate produce identical greedy completions? | `100 * full_match_rate` |
| **B — KLD@D** (KL Divergence vs reference, corpus proxy) | Mean KL divergence between candidate and reference distributions, via `llama-perplexity --kl-divergence`. | `100 * exp(-mean_kld_nats)` |

Composite: `REFRACT = harmonic_mean(GTM, KLD)` clipped to `[0, 100]`.

Bands: `[90,100]` EXCELLENT · `[80,90)` PASS · `[60,80)` DEGRADED · `[0,60)` FAIL.

## Why this exists

The paper shows that on gemma-4 26B-A4B-Q8 at q8/turbo4 KV, corpus PPL on
wikitext-2 says rotation OFF "wins" by 42% — but KLD vs the fp16-KV reference
says the *same* configuration is 1.7 nats away from fp16, the largest drift on
the row. The §4.5 noise floor on the KLD codepath is bit-exact zero on Metal,
so the signal is real. PPL is reading mis-calibration as improvement.

REFRACT rejects the PPL framing entirely: nothing matters except how close
the quantized model's behaviour stays to its fp16 self.

## Install

REFRACT is part of the `turboquant` package. From the repo root:

```bash
pip install -e .
```

Stdlib + the existing `numpy`/`scipy` deps; no new pip requirements.

## Required external binaries

REFRACT shells out to `llama-cli` and `llama-perplexity`. By default it
looks in `~/local_llms/llama.cpp/build-test/bin`. Override:

```bash
export LLAMA_CPP_BIN_DIR=/path/to/llama.cpp/build/bin
```

## Quickstart

```bash
python3 -m refract.cli score \
    --model ~/local_llms/models/Qwen2.5-7B-Instruct-Q8_0.gguf \
    --reference 'ctk=f16,ctv=f16' \
    --candidate 'ctk=q8_0,ctv=turbo4,attn_rot_v=0' \
    --prompts refract/prompts/v0.1.jsonl \
    --corpus ~/local_llms/llama.cpp/wikitext-2-raw/wiki.test.raw \
    --chunks 32 -c 512 -ngl 99 \
    --measure-floor \
    --json-out refract.qwen.json
```

Sample output (illustrative — real numbers depend on the model and build):

```
========================================================================
 REFRACT v0.1 — Reference-anchored Robust Acid-test
========================================================================
 model     : Qwen2.5-7B-Instruct-Q8_0.gguf
 reference : ctk=f16,ctv=f16
 candidate : ctk=q8_0,ctv=turbo4,attn_rot_v=0
 timestamp : 2026-04-29T11:14:02
------------------------------------------------------------------------
 Noise floor (ref vs ref):  99.99  (min 99.5)  [OK]
------------------------------------------------------------------------
 REFRACT     :  92.40  [######################################--]  EXCELLENT

 Axis A GTM  :  93.33  [######################################--]
 Axis B KLD  :  91.50  [#####################################---]
------------------------------------------------------------------------
 GTM diagnostics
   prompts                    : 30
   tokens decoded each        : 128
   full match rate            :  93.3 %
   median first divergence    : token 47
   mean prefix agreement      : 121.4 tokens

 KLD diagnostics
   chunks x ctx               : 32 x 512
   mean KLD (nats)            : 0.088732
   candidate PPL              : 6.146
   RMS Δp (vs reference)      : 3.13 %
   same top-p (vs reference)  : 95.59 %
========================================================================
```

## CLI flags

| Flag | Default | Notes |
|------|---------|-------|
| `--model PATH`        | required | GGUF path |
| `--reference SPEC`    | `ctk=f16,ctv=f16` | KV config string (see below) |
| `--candidate SPEC`    | required | KV config string |
| `--prompts PATH`      | required | JSONL prompts file |
| `--corpus PATH`       | required | plain-text corpus for KLD |
| `--chunks N`          | 32 | `--chunks` for `llama-perplexity` |
| `-c N`                | 512 | context size |
| `-ngl N`              | 99 | GPU layers |
| `--n-predict N`       | 128 | tokens per GTM prompt |
| `--seed N`            | 42 | greedy seed |
| `--measure-floor`     | off | Run REFRACT(ref, ref) and abort if < 99.5 |
| `--json-out PATH`     | — | Write the full report as JSON |
| `--no-progress`       | off | Suppress per-prompt progress lines |

### KV config spec format

`key=value,key=value,...`

| Key | Effect |
|-----|--------|
| `ctk`              | `-ctk <val>` (e.g. `f16`, `q8_0`, `turbo4`) |
| `ctv`              | `-ctv <val>` |
| `attn_rot_k`       | `LLAMA_ATTN_ROT_K_OVERRIDE=<val>` |
| `attn_rot_v`       | `LLAMA_ATTN_ROT_V_OVERRIDE=<val>` |
| `attn_rot_disable` | `LLAMA_ATTN_ROT_DISABLE=<val>` (hard lockout) |
| any other          | passed through as `--<key> <value>` |

## Validation

`tests/test_validation.py` reproduces the gemma-4 26B-A4B q8/turbo4 OFF cell
from the paper and asserts that REFRACT lands in `DEGRADED` or `FAIL`. PPL
on this cell would say "win" (-42%); REFRACT must say "audit" or "fail".

It is marked `integration` because it spawns llama.cpp many times against a
26B model. To run:

```bash
# Mark it in pytest.ini if not already:
#   markers = integration: marks tests requiring llama.cpp + a real GGUF
pytest -m integration refract/tests/test_validation.py -s
```

Or directly via CLI:

```bash
python3 -m refract.cli score \
    --model ~/local_llms/models/gemma-4-26B-A4B-Q8_0.gguf \
    --reference 'ctk=f16,ctv=f16' \
    --candidate 'ctk=q8_0,ctv=turbo4,attn_rot_v=0,attn_rot_k=0' \
    --prompts refract/prompts/v0.1.jsonl \
    --corpus ~/local_llms/llama.cpp/wikitext-2-raw/wiki.test.raw \
    --chunks 32 --measure-floor
```

Fast unit tests (no subprocess) live in `tests/test_unit.py`.

## File layout

```
refract/
  __init__.py
  cli.py                    # CLI entry point
  score.py                  # composite + bands + floor logic
  axes/
    __init__.py
    gtm.py                  # Axis A: greedy trajectory match
    kld.py                  # Axis B: KLD via llama-perplexity
  runner.py                 # llama.cpp subprocess wrappers + KVConfig
  report.py                 # text + JSON report formatter
  prompts/
    v0.1.jsonl              # 30 CC0 prompts
  tests/
    test_unit.py            # math + parser tests, no subprocess
    test_validation.py      # paper-cell reproduction (integration)
  README.md
```

## v0.2 roadmap (deferred — explicitly NOT in v0.1)

- **Axis C — R-NIAH:** retrieval needle-in-a-haystack at long context. The
  paper ends with "a more rigorous downstream probe... would likely surface
  degradation that simple factual completions hide" — R-NIAH is that probe.
- **Axis D — PLAD:** Perturbation-Locality Aware Drift. Measures whether
  drift is concentrated on specific layers or token positions, which the
  paper §3.6 nrot-ablation hints is mechanistically informative.
- **True trajectory KLD:** capture per-step logits during generation in
  GTM's forward pass instead of using corpus-KLD as a proxy. Requires a
  small llama.cpp change (or a custom binary fork) to dump logits.
- **Token-level diff via `llama-tokenize`** instead of whitespace split, so
  GTM diff positions match real BPE/SP token indices.

## Known limitations of v0.1

- KLD axis is a corpus proxy, not true trajectory KLD. Acknowledged TODO in
  `axes/kld.py`.
- GTM diff is whitespace-tokenised. Acknowledged TODO in `axes/gtm.py`.
- We strip llama-cli noise patterns by regex; new noise lines from a future
  llama.cpp release would slip through. Add to `_NOISE_PATTERNS` in
  `runner.py` if you see them.
- Floor is checked at composite level (≥ 99.5). Per-axis floor is implied
  but not separately reported. Consider adding for v0.2.
