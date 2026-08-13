# TriAttention V3 — Hybrid Model Recipe

**James Tervit, Chronara Group**

Response to the open question in triattention-v3.md Section 5:

> "On the Qwen3.5 hybrid Mamba+Attention architectures, perplexity transfers cleanly but needle retrieval silently fails at middle and end positions even under V3. The final section requests community input on the missing pieces."

This document proposes two targeted fixes and a validation recipe.

---

## Diagnosis: Why Hybrid Models Fail

The NIAH failure on Qwen3.5-27B is not a bug in V3 — it's a calibration problem. V3 applies the same eviction pressure (10%) to a model where each KV token carries 4x more retrieval weight.

### The math

| Model | Attention layers | Total layers | Attention fraction | KV per-token criticality |
|-------|-----------------|-------------|-------------------|-------------------------|
| Qwen2.5-7B | 32 | 32 | 100% | 1x (baseline) |
| Qwen3.5-27B | 16 | 64 | 25% | **4x** |
| Qwen3.5-35B-A3B | 10 | 40 | 25% | **4x** |

On Qwen2.5-7B, evicting 10% of tokens removes 10% of the model's ability to attend to that information. The other 90% of attention layers still see the remaining tokens — redundancy is high.

On Qwen3.5-27B, evicting 10% of tokens removes 10% of the information from only 16 attention layers. But those 16 layers are the **only** mechanism the model has for position-dependent retrieval (Mamba layers are position-agnostic by design). Each evicted token is a 4x larger hole in the attention fabric.

This explains every observation in the V3 paper:
- **PPL is fine** because perplexity is a near-token metric dominated by Mamba layers, which aren't affected by KV eviction
- **NIAH fails at middle/end** because those positions depend on long-range attention retrieval through the sparse attention layers
- **Start position passes** because the prefix protection (128 tokens) saves the needle
- **Boundary skip doesn't help** because the problem isn't which layers contribute to scoring — it's that too many tokens are being evicted from too few attention layers

---

## Fix 1: Scale Budget by Attention Fraction

The eviction budget should be proportional to the model's attention density, not a fixed percentage.

### Formula

```
attention_fraction = n_attention_layers / n_total_layers
eviction_rate = (1.0 - raw_budget) * attention_fraction
effective_budget = 1.0 - eviction_rate
```

### Concrete values

| Model | raw_budget | attention_fraction | effective_budget | Tokens evicted |
|-------|-----------|-------------------|-----------------|----------------|
| Qwen2.5-7B | 0.90 | 1.00 | **0.90** (10% eviction) | unchanged |
| Qwen3.5-27B | 0.90 | 0.25 | **0.975** (2.5% eviction) | 4x fewer |
| Qwen3.5-35B-A3B | 0.90 | 0.25 | **0.975** (2.5% eviction) | 4x fewer |

### Implementation sketch (in llama-triattention.cpp)

```c
// In triattention_evict(), before computing n_to_evict:

float attention_fraction = 1.0f;
if (model->hparams.ssm_d_state > 0) {
    // Hybrid model — count attention vs total layers
    int n_attn = 0;
    for (int il = 0; il < model->hparams.n_layer; il++) {
        if (is_attention_layer(model, il)) n_attn++;
    }
    attention_fraction = (float)n_attn / (float)model->hparams.n_layer;
}

float effective_evict_rate = evict_rate * attention_fraction;
int n_to_evict = (int)(n_candidates * effective_evict_rate);
```

The `is_attention_layer()` check can use `full_attention_interval` from the model config — on Qwen3.5, attention layers are at indices where `il % full_attention_interval == 0`.

### Why 2.5% eviction is still useful

At 2.5% eviction × 4.6x TurboQuant compression:
- 32K context: saves ~800 tokens of KV + 4.6x compression on the rest
- On reasoning workloads with thinking traces: the redundant `<think>` tokens score lowest and are preferentially evicted — the 2.5% that gets evicted is the 2.5% that matters least
- Combined with TQBridge: even 2.5% fewer tokens per transfer adds up across 32 layers × thousands of tokens

---

## Fix 2: Partial RoPE Frequency Count

Qwen3.5 uses partial RoPE — only `n_rot` dimensions (64 out of 256) have rotary position embeddings. The remaining 192 dimensions have no position encoding.

The trig scoring formula computes a phase-alignment score across frequency bins:

```
score = sum_f (A_f * cos_sum_f - B_f * sin_sum_f)
```

where `f` ranges over `head_dim / 2 = 128` frequency bins.

**The problem**: 96 of those 128 bins contribute zero signal because the corresponding dimensions have no RoPE rotation. The score averages 32 bins of real signal with 96 bins of noise, reducing the signal-to-noise ratio by 4x.

### Fix

```c
// Current (in triattention scoring loop):
int freq_count = head_dim / 2;  // = 128

// Fixed:
int n_rot = model->hparams.n_rot;  // = 64 for Qwen3.5, = head_dim for standard
int freq_count = n_rot / 2;         // = 32 for Qwen3.5, = 64 for standard
```

This makes the scoring 4x less noisy on Qwen3.5 without affecting standard transformers where `n_rot == head_dim`.

---

## Validation Recipe

Run these tests in order. Each builds on the previous result.

### Step 1: Apply Fix 1 only (budget scaling)

```bash
# Qwen3.5-27B, 32K context, V3 with scaled budget
# The effective budget at attention_fraction=0.25 should be ~0.975

./build-test/bin/llama-perplexity \
    -m Qwen3.5-27B-Q8_0.gguf \
    -f wikitext-2-raw/wiki.test.raw \
    -b 512 --chunks 3 -c 32768 \
    --triatt-budget 31457  # 32768 * 0.96 ≈ 31457

# Expected: PPL ≈ 7.47 (same as current V3)
```

Then NIAH:
```bash
# Middle position (65000 chars) — the failing case
./build-test/bin/llama-completion \
    -m Qwen3.5-27B-Q8_0.gguf \
    -f niah_prompt_mid.txt \
    -n 1024 -c 32768 --temp 0 -no-cnv --no-display-prompt \
    --triatt-budget 31457 --triatt-hybrid 2 --triatt-prefix 128

# Expected: PASS (or at least PARTIAL instead of FAIL)
```

### Step 2: Apply Fix 2 only (partial RoPE)

Keep the original 90% budget but fix the frequency count. This tests whether the noise reduction alone recovers NIAH.

### Step 3: Apply both fixes

The expected best result. Both fixes are orthogonal — budget scaling reduces eviction pressure, frequency fix improves eviction quality.

### Step 4: Stack with TurboQuant+

```bash
# Full stack: TQ+ (q8_0 K + turbo3 V) + V3 (scaled budget + partial RoPE fix)
./build-test/bin/llama-perplexity \
    -m Qwen3.5-27B-Q8_0.gguf \
    -f wikitext-2-raw/wiki.test.raw \
    -b 512 --chunks 3 -c 32768 \
    -ctk q8_0 -ctv turbo3 \
    --triatt-budget 31457 --triatt-hybrid 2 --triatt-prefix 128

# Expected: PPL within +1% of f16 baseline
```

---

## For TQBridge Integration

If both fixes work, the combined compression for distributed inference:

| Workload | TurboQuant | TriAttention V3 | Combined | Per-token over wire |
|----------|-----------|-----------------|----------|---------------------|
| Standard 7B, 32K | 4.6x | 1.11x (90%) | 5.1x | ~10KB |
| Hybrid 27B, 32K | 4.6x | 1.03x (97.5%) | 4.7x | ~11KB |
| Reasoning 7B, 32K | 4.6x | ~5x (est.) | ~23x | ~2.2KB |

The reasoning case is where TQBridge + TriAttention V3 delivers the most value — thinking traces generate thousands of redundant tokens that TriAttention evicts before TQBridge compresses and transfers. At 2.2KB per token, a 27B model's KV cache transfers comfortably over WiFi.

---

## Summary

Two fixes, both derived from the model architecture rather than tuning:

1. **Scale eviction by attention fraction** — fewer attention layers means each token is more critical. Don't evict 10% when each token does 4x the work.

2. **Fix frequency count for partial RoPE** — don't average 32 bins of signal with 96 bins of noise.

Neither fix requires changes to the scoring formula itself. V3's trig scoring is correct — it just needs the right inputs (frequency count) and the right budget (scaled to attention density).
