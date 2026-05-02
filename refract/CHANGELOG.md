# REFRACT changelog

Reverse-chronological. Each entry: what changed, why it changed, and the
matrix result that motivated or validated the change.

---

## v0.3.2.1 — vLLM and SGLang backends production (2026-05-02)

### What changed

- **vLLM backend rewritten from skeleton to working implementation**
  (`refract/backends/vllm.py`). Cached in-process LLM with
  evict-on-key-change for memory-pressured deployments (hybrid models
  whose two LLM instances don't fit a single accelerator). KV-config
  translation for `kv_cache_dtype` (`auto`, `fp8_e4m3`, `bfloat16`),
  greedy decode trajectories via `SamplingParams`, KLD via
  `prompt_logprobs`. Env knobs: `REFRACT_VLLM_GPU_MEMORY_UTILIZATION`,
  `REFRACT_VLLM_MAX_NUM_SEQS`, `REFRACT_VLLM_KLD_TOPK`,
  `REFRACT_VLLM_MAX_MODEL_LEN`.

- **SGLang backend new** (`refract/backends/sglang.py`). HTTP-based —
  the user runs an SGLang server separately (typically via the
  published Docker image) and REFRACT posts to it. Single-server mode
  (`REFRACT_SGLANG_URL`) for completion / trajectory / tokenize, plus
  dual-server mode (`REFRACT_SGLANG_REF_URL` + `REFRACT_SGLANG_CAND_URL`)
  for KLD which needs both KV configs simultaneously. KV dtype is
  fixed at SGLang server-launch time, so two-server (or two-phase)
  orchestration is required for cross-config comparison.

- **Trajectory and PLAD axes now batch ref/cand by KV config.**
  Previously interleaved per-prompt; now collects all ref completions
  first, then all cand completions, then computes diff/drift. Two
  model loads total per axis instead of 2N. Critical for vLLM on
  hybrid models where each load takes 30–90s.

- **Runner `tokenize_to_ids` dispatches through the active backend.**
  Previously always shelled out to `llama-tokenize`; now uses the
  backend's own tokenizer when the backend is not llamacpp. Unblocks
  R-NIAH on vLLM/SGLang and on hosts where the local llama.cpp
  checkout has drifted.

- **New extras**: `pip install -e .[refract-vllm]` (vllm),
  `pip install -e .[refract-sglang]` (requests + transformers).

### Verified end-to-end

Cross-engine bench on AMD MI300X / ROCm 7.2 (Qwen3.6-35B-A3B hybrid GDN
+ attention MoE), all four REFRACT axes (Trajectory + KLD + R-NIAH +
PLAD) on llama.cpp, vLLM, and SGLang. See
[`docs/papers/cross-engine-mi300x.md`](../docs/papers/cross-engine-mi300x.md)
for the full writeup including engine-side issues encountered (8
nontrivial bugs across the 3 engines) and reproducibility scripts.

---

## v0.3.2 — HTML reports + repeatability subcommand (2026-04-30)

### What changed

- **`refract score --html-out path.html`** — single self-contained HTML
  page with inline CSS, no JS framework, no CDN. Renders the same data
  as the JSON report plus:
  - Composite + band as a colored card
  - Diagnosis block as a colored callout
  - Per-axis bars (proper progress bars, not ASCII)
  - **R-NIAH per-(length, position) heatmap** — green=match,
    red=degraded, gray=base also fails (uninformative)
  - **PLAD per-perturbation bars** with skipped-perturbation
    callouts
  - **Run details cards**: Model (size, architecture, head count, vocab
    from GGUF / config.json), Hardware (chip, RAM, OS, NVIDIA GPUs if
    present), Environment (backend, llama.cpp commit, mlx-lm version)
  - **Repro command block** — exactly the shell-escaped argv that
    produced this report. Paste into a terminal to rerun.
  - Raw JSON embedded in a `<details>` at the bottom (HTML is the
    superset; one artifact carries everything).

  Implementation: `refract/report_html.py`, ~400 LOC. Module is
  importable independently if a script wants HTML without the rest of
  the CLI.

- **`refract repeatability` subcommand** — runs the same scoring
  config N times back-to-back, prints min/median/max/stdev/range for
  the composite and each axis, and a health verdict:
  - `✓ HEALTHY` — composite stdev ≤ 1.0
  - `⚠ NOISY` — composite stdev 1.0–3.0 (jitter present but
    rankings hold)
  - `✗ UNSTABLE` — composite stdev > 3.0 (engine non-determinism)

  Smoke-tested: 2-run on qwen2.5-1.5b-instruct q4_K_M (candidate
  q8/q8) produced stdev = 0.00 on composite, Trajectory, and KLD —
  bit-exact reproducibility on the cheap axes. R-NIAH / PLAD will
  have small variance from single-trial cells and perturbation
  RNG; run with `--full` to measure.

### Test count

82 → 82 (no new tests; HTML/repeatability tested via smoke runs).

---

## v0.3.1 — friend-ready release (2026-04-30)

### What changed

End-to-end push to make REFRACT runnable by someone other than the
author. Concrete deliverables:

- **Backend abstraction** at `refract/backends/`. `Backend` ABC defines
  the five primitives axes need (run_completion,
  run_completion_trajectory, run_kld, tokenize_to_ids,
  detect_thinking_mode). Implementations:
  - `LlamaCppBackend` — production, thin adapter over the existing
    `runner.run_*` subprocess wrappers (preserves test mocks).
  - `MLXBackend` — production, native via `mlx-lm`. Verified end-to-end
    against `Qwen2.5-3B-Instruct-4bit` (q8 KV trajectory + completion).
    KLD@D implemented natively in Python (no external binary). Known
    limitation: RotatingKVCache (gemma family) doesn't support
    quantization in mlx-lm 0.31.2.
  - `VLLMBackend` — skeleton with concrete plug-in pointers in the
    module docstring. Friends with vLLM can fill in the four methods
    against the contract.

- **Default `--axis-a` flipped from `gtm` to `trajectory`.** v0.1.x's
  GTM had a known detokenize→retokenize unit-mismatch bug that v0.1.4's
  Trajectory replaced. The default still pointed at the buggy version
  through v0.3.0 — fixed here.

- **`--backend {auto,llamacpp,mlx,vllm}` flag** routed end-to-end:
  `runner.set_active_backend()` plus dispatch in `run_completion` /
  `run_completion_trajectory` / `axes/kld.py` so a single CLI flag
  picks the inference engine for all four axes without touching axis
  code. `auto` infers from model path (.gguf → llamacpp; directory →
  mlx). Override via `REFRACT_BACKEND` env.

- **`refract selftest` subcommand.** 30-second preflight checking
  binaries, flags, env vars, and (with `--model X`) a fp16 generation
  smoke. Catches setup mistakes before the user burns 30 minutes of a
  real run finding out their llama.cpp lacks turbo. Tested end-to-end
  on llamacpp and mlx backends with real models.

- **`refract compare` subcommand.** Multi-report side-by-side table.

- **Thinking-mode auto-detection at score start-up.** Issues a tiny
  probe ("What is 2+2?"), scans for canonical thinking markers
  (`<think>`, `<|thinking|>`, gemma `<|channel|>`, etc.), prints a
  one-line transparency notice in the run banner. `Backend` exposes
  `detect_thinking_mode()` so each backend can override with a cheaper
  signal.

- **Framework version + environment metadata in JSON** (schema bumped
  to `refract.report.v0.3.1`). New top-level fields:
  `framework_version`, `environment.backend`,
  `environment.llama_cpp_commit` (when llamacpp), `environment.mlx_lm_version`
  (when mlx). Cross-person score comparison now reproducible.

- **Confidence guards on R-NIAH and PLAD axes.**
  - R-NIAH: `confidence: low` + `base_acc_avg` field when base is
    failing to engage retrieval (avg < 0.2 across cells). Avoids
    R-NIAH=100 looking like success when it's actually noise floor.
  - PLAD: `skipped_perturbations` field listing per-perturbation
    types that didn't apply (e.g., paraphrase with no synonym matches).
    `confidence: partial` when any perturbation skipped.

- **QUICKSTART.md / INTERPRETATION.md / PITFALLS.md** at the repo root.
  - QUICKSTART: 0-to-first-PASS in under 30 minutes, with selftest +
    quick + full + compare commands.
  - INTERPRETATION: per-axis "what to do if low" guide + composite
    band table + pattern-recognition Diagnosis examples.
  - PITFALLS: setup pitfalls (head_dim incompat, RotatingKVCache,
    turbo missing) + result-interpretation traps (R-NIAH refusal
    artifact, PLAD NaN) + cross-version comparison rules.

- **`examples/` directory with 4 real JSON reports** from the
  2026-04-30 matrix:
  - `clean-q8q8-mistral24b.json` (PASS, ~91)
  - `degraded-qwen7b.json` (DEGRADED, ~76)
  - `distribution-broken-gemma26b.json` (FAIL, ~29)
  - `catastrophic-symturbo.json` (FAIL, ~19; the negative control)

  Plus `examples/README.md` explaining what each shape means and
  what numbers a friend should expect to see when running similar
  configs on similar models.

### Test count

80 → 82 unit tests (+2: schema-version pin + framework-version field).

### Verified

- llamacpp backend: full v0.3.0 matrix (7 models × 4 axes) ran clean
- mlx backend: end-to-end on Qwen2.5-3B-Instruct-4bit (completion +
  trajectory with q8 KV); selftest passes with model probe
- selftest CLI: passes on llamacpp (no model) and mlx (with model)
- compare CLI: produces side-by-side table for multiple JSONs
- 82/82 unit tests pass

### Open follow-ups (still v0.3.x scope)

- vLLM backend implementation (currently skeleton; plug-in path is
  documented in module docstring)
- MLX KLD measurement on RotatingKVCache models (waits on mlx-lm)
- Cross-platform corpus shipping (currently each user fetches
  wikitext-2 themselves)
- T-Call axis (Tool-Call fidelity) — v0.4 scope, builds on v0.3 chat
  template machinery

---

## v0.3.0 — chat-template handling implemented (2026-04-30)

### What changed

`runner.run_completion()` and `runner.run_completion_trajectory()` gained
two new parameters with sensible defaults:

```python
apply_chat_template: bool = True
system: Optional[str] = None
reasoning: str = "off"           # run_completion only; llama-cli's `-rea`
```

When `apply_chat_template=True`, the underlying llama-cli / llama-completion
invocation gets `--jinja` (model's own chat template, read from GGUF),
`-rea off` (disable thinking traces — llama-cli only; llama-completion
doesn't expose this knob), and `-sys <system>` if the caller provided one.

`axes/rniah.py::_build_prompt` now returns a `(system_msg, user_msg)`
tuple. The haystack-with-needle goes into the system slot — that's where
chat templates put long-form context — and the retrieval question stays
in the user slot.

`axes/trajectory.py` and `axes/plad.py` keep their existing call sites:
they pass `apply_chat_template=True` implicitly via the new defaults, so
prompts get wrapped in the model's chat template automatically.

`KLD@D` is unchanged — it operates on natural-text corpus through
llama-perplexity, no Q&A engagement needed.

### Why mandatory not optional

See "v0.3 plan" block earlier in this file for the discovery story. Short
version: instruct-tuned models given raw `Q: ... A:` text continue
stylistically rather than answering, and emit thinking traces that burn
n_predict before the answer lands. v0.2.x R-NIAH numbers on the gemma-4
family were measured noise (base_acc = 0 across every cell) because of
this. v0.2.1 patched two symptoms (neutral needle, n_predict=256) but the
underlying issue was prompt format. v0.3.0 fixes it at the source.

### Smoke-test before kicking the matrix re-run

Same retrieval prompt run on qwen3.5-2B-Q8 fp16-vs-fp16:

  v0.2.x raw "Q: A:" with n_predict=32 → output truncated at "APRICOT-7-B"
  v0.2.1 raw "Q: A:" with n_predict=256 → keyword found inside thinking trace
  v0.3.0 chat-template `-sys ... -p ... --jinja -rea off` →
    "Based on the text provided, the rare paint color featured is
    **APRICOT-7-BLUE**. The text notes that this specific color is rare..."

Clean answer, no thinking trace, no wikitext continuation.

### Open follow-up — comparison matrix

The 7-model v0.3.0 matrix is running now (same models, same candidate
ctk=q8_0,ctv=turbo4 as v0.2.0). Results will land in a follow-up entry
below this one with the v0.2.x→v0.3.0 score deltas per model. Hypothesis:
the non-distribution axes (R-NIAH, PLAD) move materially on instruct
models that were running through raw prompts before; trajectory and KLD
shift less because they were already in a more model-native code path.

### Test count

80 → 80 (no new tests yet — v0.3.0 plumbing works through existing tests
because the new params have backward-compatible defaults). v0.3.1 should
add explicit chat-template parameter-plumbing tests.

---

## v0.3 plan — chat-template handling is MANDATORY (logged 2026-04-30)

### What we discovered during the v0.2.0 / v0.2.1 matrix runs

The R-NIAH refusal hypothesis (v0.2.1) was partially right but not the
dominant factor. The real reason 4 of 7 instruct models scored
`base_acc = 0` in every cell on v0.2.0 is the **prompt format mismatch**:

  - llama-cli was being driven with raw `Q: ... A:` text + `--single-turn`.
  - Modern instruct models (gemma-4 family, qwen3.5, gemma-E2B) expect
    their own chat template tokens (`<start_of_turn>user`, `<|im_start|>`,
    etc.) to engage Q&A mode. Given raw text in the prompt, they continue
    the wikitext stylistically instead of switching modes.
  - Some emit a thinking trace that burns the n_predict budget before any
    answer lands. qwen3.5-2B's response with the v0.2.0 default
    n_predict=32 ended at `"...the rare paint color featured in this
    article is **APRICOT-7-B"` — truncated mid-keyword.

v0.2.1 patched two of the symptoms (neutral needle, n_predict=256). The
re-run on the 4 affected models showed real engagement (base_acc up from
0% to 67%+). But the underlying issue is the prompt-format/chat-template
mismatch and that has to be fixed properly in v0.3.

### What v0.3 must do

For every axis that requires the model to engage Q&A or instruct
behaviour (Trajectory, R-NIAH, PLAD, and the proposed v0.4 T-Call axis),
the Python side has to apply the model's chat template via llama.cpp's
existing flags:

  --jinja            engine + auto-detect template from GGUF metadata
  -rea off           disable reasoning/thinking traces deterministically
  -sys "..."         system prompt (where R-NIAH context belongs)
  -p "..."           user message

`KLD@D` is the only axis that stays raw — it operates on natural-text
corpus, not Q&A.

### Implementation sketch

`runner.run_completion(...)` and `run_completion_trajectory(...)` gain
two parameters:

```python
apply_chat_template: bool = True
system: Optional[str] = None
```

When apply_chat_template is True the CLI args get `--jinja -rea off`; if
system is supplied it appends `-sys <system>`. Per-axis call sites
populate the right roles:

  Trajectory: user = each prompt; no system
  R-NIAH:     system = haystack_chunk_with_needle_inserted
              user   = retrieval question
  PLAD:       user = perturbed_prompt; no system

Estimated scope: ~80 lines of Python in runner + axes, ~20 lines of tests,
plus a full 7-model matrix re-run to compare v0.2.x scores against
v0.3.0 properly-templated scores.

### Why mandatory not optional

Without chat-template handling, results on instruct-tuned models are
*structurally* invalid for the axes that depend on Q&A engagement. The
v0.2.x R-NIAH numbers on gemma-family models are illustrative: with raw
prompts `base_acc = 0` (model never engages); with n_predict bumped and
neutral needle `base_acc > 0` (model partially engages, but via prompt
echo in thinking traces, not a real answer); with chat-template applied
(v0.3) we expect clean engagement and the score will reflect actual quant
degradation. Calling that "optional polish" understates the problem.

### What's NOT in v0.3

T-Call (tool-call fidelity) is the natural follow-on axis once
chat-template machinery exists; that's v0.4 work. Tool-spec format also
varies per-model (Qwen JSON schema, Mistral function-calling, OpenAI-
style for Gemma) so v0.4 also auto-detects spec format from
tokenizer_config.json or similar.

---

## Negative control validation — 2026-04-30

The first end-to-end matrix run (v0.2.0, 7 models, candidate `ctk=q8_0,ctv=turbo4`)
produced clean separation between healthy quants (composites 78-90) and
known-broken ones (gemma-4-26B-A4B at 29 FAIL, gemma-4-31B at 50 FAIL).
But that's a happy-path result; without a known-bad reference it could be
the framework matching whatever pattern looks "right" by coincidence. So
we ran a **deliberate negative control**:

  Model:     gemma-4-26B-A4B-it-Q8_0
  Reference: ctk=f16,ctv=f16
  Candidate: ctk=turbo4,ctv=turbo4   (SYMMETRIC turbo, paper says catastrophic)

Result on the same scoring framework:

| Surface     | Score  | Band  |
|-------------|--------|-------|
| Trajectory  | 3.93   | FAIL  |
| KLD@D       | 11.84  | FAIL  (mean_kld = 2.133 nats) |
| R-NIAH      | 100.00 | EXCELLENT (real signal; needle wording fixed in v0.2.1) |
| PLAD        | 72.21  | DEGRADED (paraphrase NaN -> partial confidence) |
| **Composite** | **~11** | **FAIL** |

Comparison on the same model, asymmetric vs symmetric turbo:

| Candidate                  | Composite | KLD nats | Trajectory |
|---------------------------|-----------|----------|------------|
| ctk=q8_0,ctv=turbo4 (asym) | 29 FAIL   | 1.738    | 17.32 |
| ctk=turbo4,ctv=turbo4 (sym)| 11 FAIL   | 2.133    | 3.93  |

Symmetric is empirically worse than asymmetric on every distribution-level
surface, which matches the paper's analytical result (rotation collisions
on both K and V destroy per-head distribution geometry). REFRACT
distinguishes the two correctly.

**Conclusion:** the framework's per-axis crater pattern + composite FAIL
behaviour on a known-catastrophic config is consistent with the paper's
predictions, so a clean PASS in the matrix is a real signal rather than a
specification artifact.

---

## v0.2.1 — 2026-04-30 (R-NIAH default needle: drop the "secret password" framing)

### Context

The v0.2.0 default R-NIAH needle was `"The secret password is APRICOT-7-BLUE."`
and the question was `"Q: What is the secret password?\nA:"`. On the 7-model
matrix run from earlier today, **4 of 7 models scored base_acc = 0 in every
cell** while still landing R-NIAH = 100. That's the "candidate matched base
in every cell" path, but with base = 0 everywhere it's a noise-floor reading,
not a real measurement of long-context retrieval.

### Diagnosis

The "secret password" wording reliably triggers safety/refusal training on
RLHF'd instruct models. Affected: gemma-4 family (E2B, 26B-A4B, 31B), small
qwen instructs (qwen3.5-2B). Affected models *can* retrieve haystack facts;
they just refuse to repeat anything framed as a credential. On the same
matrix, models without aggressive refusal training (phi-4, qwen2.5-7B,
Mistral-24B, Llama-4 Scout) had base_acc > 0 across cells and produced
real signal.

### Fix

Default needle and question changed to a non-credential framing:

  needle:   "Note: APRICOT-7-BLUE is the rare paint color featured in this article."
  question: "Q: What is the rare paint color featured in this article?\nA:"

`APRICOT-7-BLUE` keyword retained for clean substring matching. "Rare paint
color" framing has zero credential association, so safety training doesn't
fire; the question stays direct enough that instruct models engage even
on raw-completion harnesses.

### What's NOT in v0.2.1

- No fix yet for the broader chat-template engagement gap (instruct models
  expecting `<start_of_turn>user / model` but getting raw `Q: A:` style).
  Tracked for v0.3 — auto-detect chat template from `tokenizer_config.json`.
- No base-confidence guard in the R-NIAH score itself. If a future model
  legitimately can't retrieve at any cell (not refusing, just incapable),
  R-NIAH = 100 is still a noise floor. v0.3: tag the result as
  "low confidence" when base_acc averaged across cells is below ~0.2.

### Related re-run

The 4 affected models from the v0.2.0 matrix run (qwen3.5-2B, gemma-4-E2B,
gemma-4-26B-A4B, gemma-4-31B) are being re-run with the new needle so we
can replace their R-NIAH cells. The other 3 (phi-4, qwen2.5-7B, Mistral)
plus the extras (Llama-4 Scout, gemma-26B symmetric-turbo) keep their
v0.2.0 R-NIAH numbers — base engaged on those.

---

## v0.2.0 — 2026-04-30 (R-NIAH and PLAD axes implemented)

### Context

v0.1.4 shipped trajectory + v0.2 skeletons. v0.2.0 implements the two
remaining axes: R-NIAH (long-context retrieval) and PLAD (perturbation
brittleness). The full four-axis composite is now live, the report
renders all four with per-axis bands and short prose, and the JSON
schema bumped to `refract.report.v0.2.0`.

### What's new

| File | Change | Why |
|---|---|---|
| `axes/rniah.py` | Skeleton replaced with full implementation. Reads a haystack corpus, estimates chars-per-token from a head sample, builds prompts with the needle inserted at the requested position fraction (snapped to the nearest sentence boundary), runs greedy completions for each cell under both KV configs, scores each cell by case-insensitive substring match of the password keyword. Aggregates per (length, position) and returns `RNIAHResult` with full per-cell breakdown. Skipped cells (length > ctx_max) are reported, not silently dropped. | Long-context retrieval surface. Catches the "scores 99 on KLD@D and still fails at 32K context" failure mode that the other axes can't see. |
| `axes/plad.py` | Skeleton replaced with full implementation. Four perturbation generators (typo / case / punct / paraphrase) with deterministic seeding per-prompt, token-level Levenshtein for drift measurement, per-(prompt, perturbation) excess drift, score = 100·exp(-α·excess_drift). Falls back gracefully when a perturbation can't apply (e.g. typo on a prompt with no ≥4-char words). | Brittleness surface. Detects models that work on the canned demo but break on real-user typos / paraphrases. |
| `score.py` | `composite_score()` extended to four axes. R-NIAH and PLAD are optional (None values dropped before harmonic mean), so v0.1 callers get the same number they used to. v0.2 callers passing all four get a 4-axis harmonic mean. | Composite scales to 4 axes without breaking the 2-axis API. |
| `report.py` | `text_report()` and `json_report()` accept optional `rniah` and `plad` results. Per-axis lines for axes C and D added when present. Per-cell R-NIAH breakdown and per-perturbation PLAD breakdown rendered as separate diagnostics blocks. JSON schema bumped to `refract.report.v0.2.0`. | Layman + techie clarity at 4-axis scale. |
| `cli.py` | New flags: `--axis-rniah`, `--rniah-haystack`, `--rniah-ctx-max`, `--rniah-lengths`, `--rniah-positions`, `--rniah-trials`, `--axis-plad`. R-NIAH requires haystack + ctx-max; PLAD reuses `--prompts`. | Opt-in axes — v0.1.x runs unchanged unless flags are set. |
| `tests/test_trajectory.py` | +16 tests covering R-NIAH and PLAD impls (boundary detection, password extraction, empty-cells path, full-match path, candidate-loses-at-long-ctx path, perturbation generators, Levenshtein, drift computation, 2-axis vs 4-axis composite). | Pure-logic + mocked-subprocess coverage; integration runs deferred to opt-in CLI smoke. |
| `tests/test_report_json_layout.py` | Schema test bumped from `v0.1.4` → `v0.2.0`. | JSON schema contract pin. |

Total: 64 → 80 unit tests (+16), all green.

### Sample 4-axis report card (synthetic numbers)

```
 REFRACT score    :  78.66  [###############################---------]  DEGRADED
 → Visible drift. Audit on your workload before deploying.

 Axis A GTM       :  72.00  [#############################-----------]  DEGRADED   Token-level agreement with the fp16 reference.
 Axis B KLD       :  98.00  [#######################################-]  EXCELLENT  Distribution-level divergence from the fp16 reference.
 Axis C R-NIAH    :  66.70  [###########################-------------]  DEGRADED   Long-context retrieval quality vs the reference.
 Axis D PLAD      :  85.00  [##################################------]  PASS       Robustness to small prompt changes vs the reference.
```

Per-cell R-NIAH diagnostics show exactly where the candidate degrades:

```
 R-NIAH diagnostics
   per-cell (length, pos) → base_acc / cand_acc / degradation:
     ( 4096, 0.50) → 1.00 / 1.00 / 0.00
     ( 8192, 0.50) → 1.00 / 1.00 / 0.00
     (16384, 0.50) → 1.00 / 0.00 / 1.00
```

Per-perturbation PLAD diagnostics show which input variations break it:

```
 PLAD diagnostics
   typo       :  95.00 EXCELLENT
   case       :  92.00 EXCELLENT
   punct      :  78.00 DEGRADED
   paraphrase :  75.00 DEGRADED
```

### What's NOT in v0.2.0

- **No matrix re-run.** v0.1.3 estimates remain in the matrix table.
  Tom-runs-it follow-up.
- **Single needle phrasing for R-NIAH** (the protocol envisioned 5 phrasing
  variants per cell for variance reduction). v0.3.0 target.
- **Synonym table for PLAD paraphrase is ~20 entries.** Probably want to
  expand in v0.2.1 once we see how often paraphrase skips on real prompts.
- **No band recalibration.** v0.1.3's 95/80/60 thresholds remain
  provisional. Recalibration belongs to a fresh matrix run with all four
  axes scored.

---

## v0.1.4 — 2026-04-30 (trajectory axis + v0.2 skeletons)

### Context

v0.1.3 closed the *score* hole that detokenize→retokenize inflation opened
(by normalizing GTM by `mean_cand_length` instead of `n_predict`), but the
*signal* hole stayed open: comparing two retokenized texts is fundamentally
lossy when one detokenizes a 50-token model output into text that
retokenizes to 137 tokens (gemma-4 31B's 2.87× factor). v0.1.3 left a
TODO at axes/gtm.py:29-30 for v0.2 to capture token IDs at decode time
via a custom binary; v0.1.4 lands that fix.

This release also stages v0.2 by adding skeleton modules + locked-down
data shapes for R-NIAH and PLAD, the two remaining axes. Skeletons raise
`NotImplementedError` from the `run_*` entry points so callers can write
reporting code today against the v0.2 result types without waiting on
the implementations.

### What's new

| File | Change | Why |
|---|---|---|
| `tools/completion/completion.cpp` (llama.cpp fork) | Surgical patch: when `REFRACT_TRAJECTORY=<path>` env var is set, write one JSONL record per sampled token (`{"step":N,"token_id":ID}`) right after `common_sampler_sample`. ~12 lines, no new CLI flag. | Decode-time token ID capture, no detokenize round-trip. Off by default; trip-wire on env var so existing users see no behavior change. |
| `axes/trajectory.py` (NEW) | `run_trajectory()` mirroring `run_gtm()`'s signature and result shape. Drives the patched `llama-completion` binary, reads back the JSONL, computes `score = 100 * mean_prefix / mean_cand` in true model-token units. | The v0.1.4 GTM replacement. Drop-in: `axes.trajectory.run_trajectory` returns the same fields as `axes.gtm.run_gtm` so `score.composite_score` and the report layer don't change. |
| `runner.py` | New `run_completion_trajectory()` wrapper that opens a tmpfile, sets `REFRACT_TRAJECTORY`, runs `llama-completion`, parses the JSONL, deletes the tmpfile. Raises with a clear message if the patched binary isn't present (empty trajectory file). | Subprocess plumbing for the new axis, isolated from existing `run_completion()` (which still drives `llama-cli` for back-compat). |
| `cli.py` | `--axis-a {gtm,trajectory}` flag, default `gtm` for back-compat. When `trajectory`, `run_trajectory` replaces `run_gtm` for the candidate run; floor measurement still uses GTM (unchanged). | Opt-in upgrade path. Existing matrix scripts keep working; users add `--axis-a trajectory` when they have the patched binary built. |
| `axes/rniah.py` (NEW skeleton) | `run_rniah()` raises NotImplementedError. Module docstring pins the protocol: 5 lengths × 3 positions × 5 trials, needle = "APRICOT-7-BLUE", quant degradation per (length, position) cell, R-NIAH = 100·(1 - mean_cell_degradation). `RNIAHCell` and `RNIAHResult` dataclasses lock the result shape. | Long-context retrieval surface. Catches quants that score 99 on KLD@D and still fail at 32K+ context. v0.2 implementation target. |
| `axes/plad.py` (NEW skeleton) | `run_plad()` raises NotImplementedError. Module docstring pins protocol: 4 perturbation types (typo, case, punct, paraphrase), per-prompt drift comparison, score = exp(-α · excess_drift), α=5.0. `PLADPerPrompt` and `PLADResult` dataclasses lock the result shape. | Quant brittleness surface. Detects models that work on the canned demo but break on real-user typos. v0.2 implementation target. |
| `axes/__init__.py` | Status doc: gtm DEPRECATED but kept, trajectory NEW (v0.1.4), kld unchanged, rniah/plad SKELETONS. | Discoverable status without grepping. |
| `tests/test_trajectory.py` (NEW, 17 tests) | `_diff` pure-logic, `_load_prompts`, `TrajectoryResult` shape compat with `GTMResult`, mock-subprocess happy-path, empty-trajectory protective error, v0.2 skeleton importability + `NotImplementedError` + dataclass-shape pins. | All paths the new axis can exercise without firing up llama.cpp. The v0.2 shape pins are the contract a future implementer will write against. |

Total: 47 → 64 unit tests (+17), all green.

### Cosmetics: layman-friendly report cards

`text_report()` and `json_report()` v0.1.4 additions:

  - **Layman summary line** under the composite score. Maps band →
    plain English: EXCELLENT/PASS/DEGRADED/FAIL → "Indistinguishable",
    "Minor drift; safe to deploy", "Audit on your workload", "Treat as
    broken". Lets a non-techie reader take action without reading the
    paper.
  - **Per-axis bands** (e.g. `Axis A GTM: 72.00 DEGRADED`). Previously
    only the composite carried a band label; now each axis does, so a
    user sees *which* axis dragged the composite down at a glance.
  - **Per-axis prose** ("Token-level agreement with the fp16 reference",
    "Distribution-level divergence …"). Tells the reader what each
    axis actually measures.
  - **JSON schema bumped to `refract.report.v0.1.4`**. Adds top-level
    `summary`, per-axis `band`, per-axis `description`. Three new
    regression tests pin those fields so downstream pipelines can rely
    on them.

The aggregator weighting question: kept equal-weight harmonic mean as
default (the harmonic mean's "any bad axis dominates" property already
gives us the fail-loud composite). Per-use-case weighted profiles
("--profile rag" / "--profile chat") were considered and explicitly
declined — they move the framework from oracle to advisor and would
rot faster than the measurement does. If we ever weight, weight by
*measurement reliability* (KLD bit-exact > heuristic R-NIAH), not by
*use-case importance*.

### What's NOT in v0.1.4

- **No matrix re-run.** The v0.1.4 axis is implemented and smoke-tested
  against qwen2.5-1.5b (f16-vs-q8_0 KV → 100/100, full match, expected),
  but the 7-model matrix from v0.1.3 hasn't been re-run with the new
  axis. That's a Tom-runs-it follow-up — clean numbers will replace the
  v0.1.3 estimates in the matrix table.
- **No band recalibration.** v0.1.3's 95/80/60 thresholds remain
  provisional. Recalibration belongs to the v0.1.4 matrix re-run.
- **No trajectory-KLD.** v0.1.4 only captures token IDs (replaces GTM's
  text comparison). Per-step logit dump and trajectory-KLD as a separate
  axis is v0.2 work; the patch site in `tools/completion/completion.cpp`
  is well-positioned for that extension (just dump logits next to the
  token ID).

### Smoke test (qwen2.5-1.5b f16-vs-q8_0 KV)

```
score: 100.00, full_match_rate: 1.0, median_first_div: None
mean_prefix=12.0, mean_cand=12.0, mean_ref=12.0
p1 ref=[12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315]
p1 cand=[12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315]
```

Token IDs are the model's actual sampled IDs (no detokenize, no
retokenize). 12 tokens both sides, identical — q8_0 KV is bit-noise-free
vs f16 on this small model, expected.

---

## v0.1.3 — 2026-04-29 (matrix-3 → defensive patch round)

### Context

v0.1.3 is purely defensive — no new functional axes, only test coverage,
fail-loud guards, and honest docs. Triggered by:

1. **The v0.1.2 matrix run revealed a fourth GTM bug.** Even with
   model-token tokenization (the v0.1.2 fix), `mean_prefix_agreement`
   for gemma-4 31B came out at 137 tokens vs `n_predict=48` — a 2.87×
   inflation that re-clipped the GTM score to 100 on a known-degraded
   config. Root cause: detokenize → re-tokenize on verbose
   chain-of-thought outputs can produce ~3× the original token count.
2. **A codex review surfaced a silent fallback** in `axes/gtm.py`. If
   `tokenize_to_ids` raised, a bare `try/except` fell back to
   `_tokenize_words` (whitespace) — silently mixing whitespace-token
   counts with model-token expectations. The comment claimed it was
   "logged as a per-prompt note"; it wasn't. There was zero way to
   detect a wrong-unit score downstream.
3. **No regression test coverage existed** for any of the four bugs
   v0.1.x had to chase: banner-comparing (v0.1), nested-composite JSON
   schema (v0.1), backspace-noise stripping (v0.1.1), Mistral
   UnicodeDecodeError (v0.1.2), GTM unit mismatch (v0.1.2). Every patch
   round was discovered after a multi-hour matrix run.

### v0.1.2 matrix results

7 models on `q8/turbo4 OFF` vs `fp16-KV`, 30 prompts, n_predict=48,
KLD chunks=32, ctx=512.

| Model | composite | band | GTM | KLD@D | mean_kld | prefix/n_predict |
|---|---|---|---|---|---|---|
| gemma-4 31B Q8 | (clipped) | — | **100.00** ⚠ (137 tok / 48 n_pred = 2.87×) | 49.23 | 0.7086 | 2.87 ⚠ |
| phi-4 Q8 | ~65 | DEGRADED | meaningful | 99.55 | 0.0046 | <1 |
| qwen3.5-2B Q8 | ~63 | DEGRADED | meaningful | 98.35 | 0.0167 | <1 |
| gemma-4 E2B Q4_K_L | ~60 | FAIL | meaningful | 93.50 | 0.0672 | <1 |
| qwen2.5-7B Q8 | ~44 | FAIL | meaningful | 98.75 | 0.0126 | <1 |
| gemma-4 26B-A4B Q8 | ~25 | FAIL | meaningful | 17.59 | 1.7381 | <1 |
| Mistral-Small-24B Q4 | ran | — | meaningful | parsed (no crash, errors='replace' fix) | — | — |

What v0.1.2 fixed (still good in v0.1.3):
- KLD@D rankings still match paper §4.3 perfectly.
- Mistral no longer crashes with `UnicodeDecodeError` (errors='replace'
  fix worked).
- 6 of 7 models produced meaningful, non-clipped GTM scores in the
  right units.

What v0.1.2 didn't fix:
- gemma-4 31B re-clipped GTM=100 due to retokenize inflation. The
  fix in v0.1.3 is to normalize by `mean_cand_length` instead of
  `n_predict`.

### v0.1.3 changes

| File | Change | Why |
|---|---|---|
| `axes/gtm.py` | Tokenizer-failure fallback now RAISES instead of silently using whitespace | The previous `try/except → _tokenize_words` produced wrong-unit scores with no way to detect. Fail-loud is correct here — a wrong score is worse than no score. |
| `axes/gtm.py` | Score now `100 * mean_prefix / mean_cand_length` (was `... / n_predict`) | Bounded in [0, 1] regardless of detokenize→retokenize inflation. Fixes the gemma-31B 2.87× clip. |
| `axes/gtm.py` | `GTMResult` exposes `mean_cand_length`, `mean_ref_length`, `notes` | So aggregators can spot retokenize inflation directly. |
| `cli.py` | `--measure-floor` additionally asserts `mean_prefix == mean_cand_length` on ref-vs-ref | KLD ref-vs-ref is bit-exact zero on Metal so the composite floor passes even when GTM is broken. The new check catches that. |
| `runner.py` | `_TOPP_RE` regex tolerates `Same top p` (space) and `Same top-p` (dash) | A test surfaced that the v0.1.2 regex only matched the dashed form. The space form occurs in real llama-perplexity output. |
| `runner.py` | New `corpus_identity()`, `write_corpus_sidecar()`, `read_corpus_sidecar()`, `assert_corpus_matches()` | Records corpus path/size/SHA next to the KLD base file. Refuses to score against a base built from a different corpus. |
| `axes/kld.py` | Writes a `<base>.corpus.json` sidecar, reads it on user-supplied bases, includes corpus identity in `KLDResult` | Wires the new machinery in. JSON now records what corpus the run was scored on. |
| `report.py` | Schema bumped: `refract.report.v0.1.1` → `refract.report.v0.1.3` | Both the GTMResult shape and the corpus-identity field changed; the schema number is the integration contract. |
| `report.py` | Text report shows `mean cand / ref length` and any GTM `notes` | Surfaces the retokenize-inflation diagnostic. |
| `__init__.py` | `__version__ = "0.1.3"` | Match the schema. |
| `tests/test_strip_noise.py` | NEW — 5 tests pinning `_strip_noise` against the v0.1.1 banner / backspace / block-char regressions | Would have caught the v0.1.1 banner-comparing bug pre-matrix. |
| `tests/test_command_construction.py` | NEW — 4 tests pinning `--single-turn` / no-`--no-conversation` / `--kl-divergence-base` / `errors='replace'` | Would have caught the v0.1.1 (`--no-conversation` regression) and v0.1.2 (Unicode crash) bugs pre-matrix. |
| `tests/test_tokenize_to_ids.py` | NEW — 5 tests pinning the `[1, 2, 3]` parser, empty-input short-circuit, and the v0.1.3 raise-on-error contract | Would have caught a future regression of the v0.1.3 fail-loud contract. |
| `tests/test_kld_regex.py` | NEW — 6 tests pinning the perplexity regexes, including the gemma-missing-Same-top-p edge case | Surfaced and fixed the `Same top p` (space) parsing bug while writing the test. |
| `tests/test_report_json_layout.py` | NEW — 5 tests pinning the JSON schema (composite is scalar, schema string, axes block) | Would have caught the v0.1 nested-composite serialisation bug pre-matrix. |
| `tests/test_corpus_identity.py` | NEW — 5 tests for the new sidecar machinery | Guards the corpus-mismatch check. |
| `tests/test_validation.py` | Strengthened with explicit GTM assertions: `full_match_rate < 1.0`, `mean_prefix > 0`, ≥3 distinct ref texts | Even the integration test would have flagged the v0.1.1 banner bug if it had these; previously it asserted only on `composite.band`. |
| `README.md` | Refresh to reflect v0.1.2 model-token tokenization + v0.1.3 candidate-length normalization, point to `LIMITATIONS.md`, list new test files | Docs honesty. |
| `axes/gtm.py` (module docstring) | Same docs cleanup | Docs honesty. |
| `LIMITATIONS.md` | NEW — documents the limitations we're shipping with: detokenize→retokenize gap, corpus-anchored KLD, provisional bands, single platform tested, GTM/EOS conflation, whitespace fallback removal | Codex review surfaced these explicitly; documenting them is more honest than burying TODO comments. |

### Test count

| Version | Total tests | Notes |
|---|---|---|
| v0.1 – v0.1.2 | 14 | Math + parsers only; subprocess wrappers and runner had zero coverage. |
| v0.1.3 | **44** unit + 1 integration | 30 new regression tests across 6 new files; integration test strengthened with 4 new GTM assertions. |

v0.1.3 is the first version with regression test coverage of the
subprocess and runner layers. Everything that broke in v0.1.x now has
a fast, no-subprocess test that would have caught the bug before the
matrix run.

### Open questions (deliberately deferred to v0.2)

- **Should the corpus-identity check be a hard error or a warning?**
  Current behaviour is hard error (RuntimeError). Argument for warning:
  in pipeline use, a user might intentionally re-use a base file across
  truncations of the same corpus. Argument for hard error: the cost of
  a meaningless KLD result that gets believed is much higher than the
  cost of a misfire.
- **Bands re-calibration once trajectory KLD lands** — current 95/80/60
  thresholds are calibrated against corpus-KLD and will need updating
  when the absolute KLD scale changes.

---

## v0.1.2 — 2026-04-29 (matrix-2 → matrix-3 patch round)

### Context

v0.1.1's matrix run surfaced two more bugs:

1. **Mistral-Small-24B failed in the KLD subprocess with `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc4 in position 5204: invalid continuation byte`**. `subprocess.run(..., text=True)` in `runner.py` strict-decodes stderr as utf-8. llama-perplexity occasionally emits non-utf-8 bytes (likely from a model name or progress string), which crashes the entire score before any KLD numbers are returned. Mistral failed in v0.1 too (we thought it was timeout) but the actual failure mode this time was the decode crash, not timeout.
2. **GTM unit mismatch**: `mean_prefix_agreement_length` is in *whitespace-tokens* while `n_predict` is in *model tokens*. Whitespace tokenization can produce more tokens than the model decoded (or fewer, depending on the text shape), so the score `100 * mean_prefix / n_predict` could exceed 1.0 and hit the clip. v0.1.1 matrix showed this on gemma-4 31B, where `mean_prefix = 59 (whitespace) > n_predict = 48 (model)` produced a clipped GTM = 100 even though KLD was 49.

### v0.1.1 matrix results (recorded for reference)

7 models on `q8/turbo4 OFF` vs `fp16-KV`, 30 prompts, n_predict=48, KLD chunks=32, ctx=512.

| Model | composite | band | GTM | KLD@D | mean_kld | prefix/n |
|---|---|---|---|---|---|---|
| gemma-4 31B Q8 | 65.98 | DEGRADED | **100.00** ⚠ | 49.23 | 0.7086 | 1.234 ⚠ |
| phi-4 Q8 | 65.13 | DEGRADED | 48.40 | 99.55 | 0.0046 | 0.484 |
| qwen3.5-2B Q8 | 63.11 | DEGRADED | 46.46 | 98.35 | 0.0167 | 0.465 |
| gemma-4 E2B Q4_K_L | 59.99 | FAIL | 44.17 | 93.50 | 0.0672 | 0.442 |
| qwen2.5-7B Q8 | 43.78 | FAIL | 28.12 | 98.75 | 0.0126 | 0.281 |
| gemma-4 26B-A4B Q8 | 25.01 | FAIL | 43.26 | 17.59 | 1.7381 | 0.433 |
| Mistral-Small-24B Q4 | — | — | 55.83 | UnicodeDecodeError | — | — |

What was real in v0.1.1:
- KLD@D rankings are unchanged from v0.1, still match paper §4.3 perfectly. KLD axis is the trustworthy half.
- Composite ranking correctly identifies gemma-4 26B-A4B as worst (composite 25, FAIL) and shows the gemma artifact as the dominant signal.

What was fake in v0.1.1:
- gemma-4 31B GTM = 100. Whitespace-token over-count clipped to 100. Real model-token prefix-agreement is much lower; the score is artificially inflating.
- Composite ordering of healthy models. phi-4, qwen3.5-2B, gemma-31B all land 63–66 because GTM dominates the composite and GTM is in wrong units. Once GTM is fixed, composite ordering for the healthy tail should track KLD@D.

### v0.1.2 changes

| File | Change | Why |
|---|---|---|
| `runner.py` | Add `errors="replace"` to all 3 `subprocess.run(text=True, ...)` calls | Fixes Mistral's UnicodeDecodeError. llama-perplexity stderr can contain non-utf-8 bytes from model loading; strict utf-8 decode crashes. Replace policy keeps the stream usable. |
| `runner.py` | New `tokenize_to_ids(model, text)` helper that shells to `llama-tokenize --ids --no-bos --no-parse-special --log-disable --stdin` | Enables true model-token diffing for GTM. Returns a `list[int]` of token IDs. Inherits `errors="replace"`. |
| `axes/gtm.py` | Replace `_tokenize_words` with `tokenize_to_ids` in the diff path | Fixes the unit mismatch. `mean_prefix_agreement_length` is now in model tokens, comparable to `n_predict`. The whitespace-tokenizer is kept as a fallback in case `llama-tokenize` errors. |
| (no test changes) | | `_tokenize_words` is still in unit tests as a stable utility; the replacement is in the diff path only. |

### v0.1.2 smoke test (Qwen3.5-2B Q8, 5 prompts, n_predict=32, chunks=8)

```
GTM     diagnostics
  full match rate         : 60.0 %
  median first divergence : token 20.0
  mean prefix agreement   : 30.0 tokens     (model tokens now, was 17.4 whitespace tokens)
KLD     diagnostics
  mean KLD (nats)         : 0.0158
```

GTM score: `100 * 30/32` = **93.75** (was 54.4 with whitespace tokenization). 30 of 32 model tokens matched on average — meaningful, not clipped, comparable across models.

### v0.1.2 matrix results

*(Pending; will append when the v0.1.2 7-model run completes.)*

---

## v0.1.1 — 2026-04-29 (matrix-1 → matrix-2 patch round)

### Context

v0.1's first 7-model matrix run surfaced four bugs that together made the
output unusable:

1. **Composite scalar serialised as a nested dict.** `d['composite']` returned
   `{"composite": 60.23, "band": "DEGRADED", ...}` instead of the scalar.
   Aggregator scripts that did `f"{d['composite']:.2f}"` printed
   `{'compos…` and pretended that was a number.
2. **GTM axis was comparing llama-cli help banners, not generations.** The
   runner passed `--no-conversation` to llama-cli. This fork rejects that
   flag with a custom error (`please use llama-completion instead`) and
   falls through to printing help. The runner captured the help text and
   compared two help banners against each other across all 30 prompts. Match
   rates of 33–57% were entirely artifacts of how often two help banners
   happened to match — not of model faithfulness.
3. **Backspace control characters in llama-cli output broke the noise filter.**
   This fork's llama-cli emits the loading spinner and the generation prefix
   with `\x08` (backspace) chars: `|\x08 \x08[Start thinking]`. The regex
   `^\|\s.*` did not match because `\x08` is not whitespace. Even after the
   `--no-conversation` fix, the noise filter dropped real generations.
4. **Per-model timeout was 30 minutes.** Mistral-Small-24B's full GTM + KLD
   run on a 24B Q4 model needs longer than 30 min on M5 Max. Mistral failed
   with a Python `TimeoutExpired` exception in the KLD-base subprocess.
   (Re-classified in v0.1.2: the actual root cause was UnicodeDecodeError
   on llama-perplexity stderr, not the timeout. The 30-min timeout was a
   contributing factor on the v0.1 run but not the proximate cause.)

### v0.1 matrix results (recorded for reference; numbers are partly invalid)

7 models on `q8/turbo4 OFF` vs `fp16-KV` reference, 30 prompts, n_predict=48,
KLD chunks=32, ctx=512.

| Model | GTM (binary, banner-noise) | KLD@D | mean KLD nats | Hand-computed composite | Notes |
|---|---|---|---|---|---|
| phi-4 Q8 | 56.67 | 99.55 | 0.0046 | 72.21 (DEGRADED) | KLD valid; GTM is banner |
| qwen2.5-7B Q8 | 43.33 | 98.75 | 0.0126 | 60.23 (DEGRADED) | KLD valid; GTM is banner |
| qwen3.5-2B Q8 | 33.33 | 98.35 | 0.0167 | 49.78 (FAIL) | KLD valid; GTM is banner |
| gemma-4 E2B Q4_K_L | 30.00 | 93.50 | 0.0672 | 45.42 (FAIL) | KLD valid; GTM is banner |
| gemma-4 31B Q8 | 23.33 | 49.23 | 0.7086 | 31.66 (FAIL) | KLD valid; GTM is banner |
| gemma-4 26B-A4B Q8 | 36.67 | 17.59 | 1.7381 | 23.78 (FAIL) | KLD valid; GTM is banner |
| Mistral-Small-24B Q4 | 43.33 | — | — | — | killed by 30-min outer timeout |

What was real in v0.1:
- KLD@D rankings are correct and match the paper §4.3 exactly. exp(-KLD)·100
  reproduces the full distribution-drift story (gemma-26B-A4B at 17.59,
  Qwen2.5-7B at 98.75, etc.).
- The composite ranking would have been correct *for the right reasons* if
  GTM were real, because KLD already separates the regimes.

What was fake in v0.1:
- Every GTM number. The "match rate" was the rate at which two captures of
  llama-cli's help text happened to be byte-identical (apparently ~33–57%
  depending on stochastic content like timestamps inside the help text).
- The composite score field in the JSON.

### v0.1.1 changes

| File | Change | Why |
|---|---|---|
| `runner.py` | Drop `--no-conversation` from llama-cli args | This fork rejects the flag and falls through to help-mode. `--single-turn` alone is enough to disable interactive mode. |
| `runner.py` | Strip `\x08` (backspace) from captured stdout before noise filtering | llama-cli puts backspaces inside the spinner AND the `|<BS> <BS>[…]` generation prefix. Without stripping them, the gen-line regex never matches. |
| `runner.py` | Add `Loading model…` and `> prompt-echo` to `_NOISE_PATTERNS` | Cleans up output so the gen-line regex finds the right entry point. |
| `runner.py` | Add gen-line extractor: keep only text from first `^\|\s` line, strip the `| ` prefix | Removes the build-info / available-commands banner that prints on every run on this fork. |
| `runner.py` | Add ASCII-art block-character stripper | The fork's logo uses Unicode block chars; previous filter left them. |
| `axes/gtm.py` | Switch GTM score from `100 * full_match_rate` (binary per prompt) to `100 * mean_prefix_agreement_length / n_predict` (continuous) | Binary was too strict: a single-token divergence at position 5 of 48 zeroed the prompt's contribution. Continuous version distinguishes "matched 5 tokens" from "matched 47 tokens", which is what we actually care about for ranking near-faithful quantizations. `full_match_rate` is still reported as a diagnostic. |
| `report.py` | Flatten `composite` to a scalar at the JSON top level; full breakdown moves to `composite_detail` | Aggregators expected `d['composite']` to be a number per the spec mockups. v0.1 nested it as a dict. Bumped JSON schema to `refract.report.v0.1.1`. |
| `/tmp/refract-matrix.sh` | Per-model `timeout 1800` → `timeout 7200` | Mistral-Small-24B's KLD axis needed >30 min. With real generation now (not banner echo) GTM also takes longer per call. 2 hours per model is the new safety margin. |

### v0.1.1 smoke test (Qwen3.5-2B Q8, 5 prompts, n_predict=32, chunks=8)

```
GTM     diagnostics
  full match rate         : 60.0 %     (now meaningful — different prompts give different outputs)
  median first divergence : token 10
  mean prefix agreement   : 17.4 tokens   (continuous score = 100 * 17.4/32 = 54)
KLD     diagnostics
  mean KLD (nats)         : 0.0158     (matches v0.1 KLD; KLD axis was always correct)
```

Captured ref text post-fix (truncated):
```
prompt: 'The capital of France is'
  ref: '[Start thinking]\nThinking Process:\n\n1. **Analyze the Request:** The user is asking a simple factual question: "The capital of France is".'
prompt: 'The largest ocean on Earth is the'
  ref: '[Start thinking]\nThinking Process:\n\n1. **Analyze the Request:** The user is asking a simple factual question: "The largest ocean on Earth is the".'
```

Different prompts → different outputs ✓. Banner stripped ✓. JSON `d['composite']`
is now a scalar ✓.

---

## v0.1 — 2026-04-29 (initial implementation)

First reference implementation. Two of four planned axes (GTM and KLD@D);
R-NIAH and PLAD deferred to v0.2.

Spec source: post-paper conversation in this repo. Goals: replace PPL with a
reference-anchored multi-axis composite, ship a CLI usable on any GGUF model,
verify Metal-determinism via a noise-floor measurement.

Initial files:
- `cli.py` — argparse front-end
- `runner.py` — subprocess wrappers around `llama-cli` and `llama-perplexity`
- `score.py` — harmonic-mean composite + EXCELLENT/PASS/DEGRADED/FAIL bands
- `report.py` — text + JSON report formatter
- `axes/gtm.py` — Greedy Trajectory Match (binary full-match-rate)
- `axes/kld.py` — KL Divergence at Decode (corpus-proxy via llama-perplexity)
- `prompts/v0.1.jsonl` — 30 CC0 prompts (factual / arithmetic / code / reasoning / instruction / dialogue)
- `tests/test_unit.py` — 14 unit tests
- `tests/test_validation.py` — gated integration test against gemma-4 26B-A4B

Initial bands (calibrated against v0.1.1 matrix; will be revisited in v0.2):
- 95–100 EXCELLENT
- 80–94  PASS
- 60–79  DEGRADED
- <60    FAIL
