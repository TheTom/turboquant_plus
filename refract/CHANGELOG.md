# REFRACT changelog

Reverse-chronological. Each entry: what changed, why it changed, and the
matrix result that motivated or validated the change.

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
REFRACT: <reported by full matrix run, not smoke>
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

### v0.1.1 matrix results

*(Pending the v0.1.1 full matrix run; will be appended to this section when it
completes. Same 7 models, same configs, comparable to the v0.1 table above so
the GTM-fix delta is visible.)*

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
