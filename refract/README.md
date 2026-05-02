# REFRACT v0.3.2

> **⚠️ ALPHA — for initial testing and feedback.**
> Setup is manual; proper packaging (PyPI, bundled corpus/prompts,
> a `setup.sh` for the llama.cpp build) is coming. Run via
> `python3 -m refract.cli` for now. See
> [QUICKSTART.md](QUICKSTART.md) for the 0-to-first-PASS path.

**REF**erence-anchored **R**obust **A**cid-test for **C**ompressed **T**ransformers.

## Install

```bash
pip install -e .                    # REFRACT alpha — zero non-stdlib deps
pip install -e .[refract-mlx]       # add the MLX backend (Apple Silicon)
pip install -e .[refract-vllm]      # add the vLLM backend (CUDA / ROCm)
pip install -e .[refract-sglang]    # add the SGLang backend (HTTP client + tokenizer)
pip install -e .[turboquant]        # add the TurboQuant Python implementation
pip install -e .[dev]               # pytest + coverage
```

The base install gives you the `refract` CLI with no third-party
dependencies. Backends are extras you opt into.

## Platform support

REFRACT itself (Python framework + report renderer + CLI) is
platform-portable. The constraint is which **backend** runs on your OS:

| OS | llamacpp | mlx | vllm | sglang |
|---|---|---|---|---|
| macOS (Apple Silicon) | ✅ primary dev target | ✅ production | n/a | n/a |
| Linux (Ubuntu 24.04, x86_64, ROCm 7.2 MI300X) | ✅ verified | n/a (Apple Silicon only) | ✅ verified | ✅ verified (HTTP client; SGLang server runs separately) |
| Windows | ⚠️ TBD (Python side portable; not run end-to-end yet) | n/a | ⚠️ TBD | ⚠️ TBD |

vLLM and SGLang backends were both verified end-to-end on the AMD MI300X
droplet on hybrid Qwen3.6-35B-A3B during the cross-engine bench documented
in [`docs/papers/cross-engine-mi300x.md`](../docs/papers/cross-engine-mi300x.md).

The llama.cpp backend needs the patched binaries (`llama-cli`,
`llama-completion`, `llama-tokenize`, `llama-perplexity`) on the loader
path — `LD_LIBRARY_PATH` / `ldconfig` on Linux, DLLs next to the `.exe`
or on `PATH` on Windows. `selftest` detects this and prints the right
remediation per OS. MLX is Apple Silicon only by upstream design.

The vLLM backend uses `vllm.LLM` in-process. Each call instantiates an
LLM at the requested KV config; backend caches one LLM at a time and
evicts on key change for memory-pressured deployments. Env knobs:
`REFRACT_VLLM_GPU_MEMORY_UTILIZATION`, `REFRACT_VLLM_MAX_NUM_SEQS`,
`REFRACT_VLLM_KLD_TOPK`, `REFRACT_VLLM_MAX_MODEL_LEN`.

The SGLang backend is HTTP-based — the user runs an SGLang server
separately (typically via the published Docker image), and REFRACT
posts to it. KV dtype is fixed at SGLang server-launch time, so
`run_kld` (which compares two configs) requires either two simultaneous
servers (`REFRACT_SGLANG_REF_URL` + `REFRACT_SGLANG_CAND_URL`) or a
two-phase orchestrator that launches them sequentially. See
`docs/papers/cross-engine-mi300x.md` §6 for a working orchestrator.

Friend-tester input on Windows is welcome — open an issue with your
`refract selftest` output.

## Where do I go?

| If you want to… | Read |
|---|---|
| Understand what REFRACT is and why it exists | This file (below) + [`docs/papers/attn-rotation-and-ppl-artifact.md`](../docs/papers/attn-rotation-and-ppl-artifact.md) |
| Get to a real score in 30 minutes | [QUICKSTART.md](QUICKSTART.md) |
| Read your own report (figure out what your score means) | [INTERPRETATION.md](INTERPRETATION.md) |
| See which models score how on which KV configs | [LEADERBOARD.md](LEADERBOARD.md) |
| Avoid known setup / interpretation traps | [PITFALLS.md](PITFALLS.md) |
| See what v0.3 explicitly does NOT do | [LIMITATIONS.md](LIMITATIONS.md) |
| See what changed across versions | [CHANGELOG.md](CHANGELOG.md) |
| Compare your run to known-good reference numbers | [examples/](examples/) (4 sample JSONs + HTMLs) |
| See the methodology evolution data | [MATRIX-RESULTS.md](MATRIX-RESULTS.md) |


A benchmaxx-resistant alternative to corpus PPL for evaluating KV-cache
quantization quality. Replaces "lower PPL = better" — a metric the paper
[`docs/papers/attn-rotation-and-ppl-artifact.md`](../docs/papers/attn-rotation-and-ppl-artifact.md)
shows can invert sign on instruct-tuned models — with a 4-axis composite
that ranks configurations by *distance from the fp16-KV reference*, not
by absolute corpus likelihood.

## Why this exists

The motivation paper documents a real failure of corpus PPL: on
**gemma-4-26B-A4B-Q8 with q8/turbo4 KV**, wikitext-2 PPL says rotation
OFF "wins" by 42%, but **KLD vs the fp16-KV reference says the same
configuration is 1.7 nats away from fp16** — the largest distribution
drift on the row. The KLD codepath is bit-exact zero on Metal, so the
signal is real. PPL is reading miscalibration as improvement.

REFRACT rejects the PPL framing entirely: nothing matters except how
close the quantized model's behaviour stays to its fp16 self.

Read [`docs/papers/attn-rotation-and-ppl-artifact.md`](../docs/papers/attn-rotation-and-ppl-artifact.md)
for the full motivation.

## What ships in v0.3.2

Four axes, each scored 0–100 (higher is better) against the model's own
fp16-KV reference:

| Axis | Name | What it measures | Notes |
|------|------|------------------|-------|
| A | **Trajectory** | Token-level agreement on greedy decode (decode-time IDs, no detokenize round-trip) | v0.1.4+; replaces the buggy GTM default |
| B | **KLD@D** | Distribution-level divergence on a natural-text corpus | Bit-exact zero on Metal at ref==cand |
| C | **R-NIAH** | Long-context retrieval quality (needle-in-haystack at multiple lengths/positions) | v0.2.0+; opt-in via `--full` |
| D | **PLAD** | Robustness to small prompt perturbations (typo/case/punct/paraphrase) | v0.2.0+; opt-in via `--full` |

**Composite** = harmonic mean of the axes that ran. Any single broken
axis tanks the composite — the framework is intentionally fail-loud.

**Bands**: `[90,100]` EXCELLENT · `[80,90)` PASS · `[60,80)` DEGRADED · `[0,60)` FAIL.

**Backends**: llama.cpp (production), MLX (production via mlx-lm),
vLLM (production, in-process LLM, ROCm + CUDA verified), SGLang
(production, HTTP client to a pre-launched SGLang server).

## Subcommands

```
refract score          # score a candidate KV config
refract selftest       # 30s preflight: binaries, flags, model probe
refract compare        # multi-report side-by-side
refract repeatability  # run N times, report spread (stdev/range)
refract fetch          # download wikitext-2-raw corpus to ~/.cache/refract/
```

## Reports

Every `score` run can emit two formats via `--json-out` and `--html-out`:

- **JSON** (`--json-out report.json`) — schema `refract.report.v0.3.1`,
  consumable by `refract compare` or any JSON-aware tool.
- **HTML** (`--html-out report.html`) — single **self-contained file**
  (~40 KB) with composite stats, diagnosis callout, per-axis bars,
  R-NIAH heatmap, PLAD per-perturbation table, run details (hardware +
  model + env), the sanitized repro command, and the raw JSON embedded
  in a collapsible section. Sun/moon toggle in the top-right for
  light/dark mode (follows OS by default). Pasteable in Discord/X.
  See [`examples/`](examples/) for 4 real samples.

The HTML uses `light-dark()` CSS (Chrome 123+ / Safari 17.5+ / Firefox
120+) for dark mode and Google Fonts CDN for typography (system-ui
fallback offline). Older browsers see the light theme cleanly.

## Quickstart

See [QUICKSTART.md](QUICKSTART.md) for full setup. Short version:

```bash
# 1. Verify your setup
python3 -m refract.cli selftest --backend auto --model path/to/model.gguf

# 2. First quick score (~5-7 min on a 7B Q8)
python3 -m refract.cli score \
    --model path/to/model.gguf \
    --candidate "ctk=q8_0,ctv=q8_0" \
    --prompts refract/prompts/v0.1.jsonl \
    --corpus path/to/wiki.test.raw \
    --json-out report.json \
    --html-out report.html

# 3. Full audit (~25-30 min on a 7B Q8)
python3 -m refract.cli score \
    --model path/to/model.gguf \
    --candidate "ctk=q8_0,ctv=q8_0" \
    --prompts refract/prompts/v0.1.jsonl \
    --corpus path/to/wiki.test.raw \
    --full \
    --rniah-haystack path/to/wiki.train.raw \
    --rniah-ctx-max 16384 \
    --json-out report.json --html-out report.html

# 4. Verify reproducibility (4 runs, expect stdev ≤ 1.0)
python3 -m refract.cli repeatability \
    --model path/to/model.gguf \
    --candidate "ctk=q8_0,ctv=q8_0" \
    --prompts refract/prompts/v0.1.jsonl \
    --corpus path/to/wiki.test.raw \
    --runs 4
```

## Documentation

| File | When to read |
|------|--------------|
| [QUICKSTART.md](QUICKSTART.md) | First-time setup + first run |
| [INTERPRETATION.md](INTERPRETATION.md) | What does my score mean? Per-axis "what to do if low" |
| [LEADERBOARD.md](LEADERBOARD.md) | Cross-model rankings on which KV configs (with the strong "this is NOT a model-quality leaderboard" disclaimer) |
| [PITFALLS.md](PITFALLS.md) | Things that have actually bitten us — avoid them |
| [LIMITATIONS.md](LIMITATIONS.md) | What v0.3 explicitly does NOT do |
| [CHANGELOG.md](CHANGELOG.md) | Full history including the v0.2 / v0.3 discoveries |
| [MATRIX-RESULTS.md](MATRIX-RESULTS.md) | Reference numbers from the 7-model 2026-04-30 matrix |
| [examples/](examples/) | Sample JSONs + HTML reports (clean / degraded / distribution-broken / catastrophic) |
| [docs/papers/attn-rotation-and-ppl-artifact.md](../docs/papers/attn-rotation-and-ppl-artifact.md) | Why this framework exists at all (the motivation paper) |

## File layout

```
refract/
  __init__.py             # version stamp
  cli.py                  # CLI: score / selftest / compare / repeatability
  score.py                # composite + bands + diagnosis
  report.py               # text + JSON report formatter
  report_html.py          # self-contained HTML report (v0.3.2+)
  runner.py               # llama.cpp subprocess wrappers + KVConfig
  axes/
    gtm.py                # Axis A: deprecated retokenize variant
    trajectory.py         # Axis A: v0.1.4+ decode-time token IDs
    kld.py                # Axis B: KLD via llama-perplexity / native MLX
    rniah.py              # Axis C: needle-in-haystack
    plad.py               # Axis D: perturbation drift
  backends/
    base.py               # Backend ABC
    llamacpp.py           # llama.cpp subprocess backend (production)
    mlx.py                # MLX native Python backend (production)
    vllm.py               # vLLM backend (production; CUDA / ROCm)
    sglang.py             # SGLang HTTP backend (production)
  prompts/v0.1.jsonl      # 30 CC0 prompts
  examples/               # 4 sample JSON reports + README
  tests/                  # 82 unit tests + 1 integration test
  README.md               # this file
  QUICKSTART.md           # setup + first run
  INTERPRETATION.md       # how to read a report
  PITFALLS.md             # known traps
  LIMITATIONS.md          # what v0.3 doesn't do
  CHANGELOG.md            # reverse-chronological
  MATRIX-RESULTS.md       # 2026-04-30 7-model matrix
```

## Status

  - **Production**: llama.cpp backend, all four axes, MLX backend
    (Trajectory + KLD + R-NIAH + PLAD).
  - **Skeleton**: vLLM backend (interface defined; plug-in points
    documented in `backends/vllm.py` docstring).
  - **Open**: T-Call axis (tool-call fidelity) — v0.4 target;
    multi-prompt-set support; bundled corpus distribution.

## Contributing

This is alpha. Open issues with:
  - Your `selftest` output (so we know what you have)
  - The full JSON of any failing run (`--json-out`)
  - The HTML report if you want a visual share (`--html-out`)
  - Your model + KV config

Especially valuable feedback: surfaces where REFRACT fails silently
(low base_acc, NaN perturbations, etc.) before the confidence guards
catch them.
