# Benchmarks

Full benchmark and validation data for TurboQuant+ on llama.cpp. The headline quality table lives in the [README](../README.md); everything else is here. Hardware is M5 Max 128GB unless a section says otherwise.

## Asymmetric K/V (NEW)

TurboQuant supports independent K and V cache types. In current testing, keeping K at q8_0 while compressing V with turbo rescues quality on low-bit models where symmetric turbo degrades:

| Model (weights) | K | V | PPL | vs q8_0 |
|-----------------|---|---|------|---------|
| Qwen2.5-7B (Q4_K_M) | q8_0 | turbo4 | 6.64 | +1.0% |
| Qwen2.5-7B (Q4_K_M) | q8_0 | turbo3 | 6.71 | +2.0% |
| Qwen2.5-7B (Q4_K_M) | turbo3 | turbo3 | 3556 | catastrophic |

```bash
# Validated starting point for low-bit models
# (tested on Qwen2.5-7B Q4_K_M; not all Q4_K_M models need this)
llama-server -m model-Q4_K_M.gguf -ctk q8_0 -ctv turbo4 -fa 1
```

## Boundary V (Layer-Aware V Compression)

Not all V layers need the same precision. Boundary V protects the first 2 + last 2 layers with q8_0-V while compressing all remaining layers with turbo2-V. 15 lines of code, no speed penalty.

| Model | turbo2 PPL | Boundary V PPL | turbo3 PPL | Quality recovered |
|-------|-----------|---------------|-----------|-------------------|
| phi-4-Q8_0 (40L) | 4.835 | 4.784 | 4.742 | 55% |
| Qwen2.5-7B Q4_K_M (28L) | 6.911 | 6.835 | 6.707 | 37% |
| Qwen3.5-35B MoE (64L) | 5.257 | 5.148 | 5.137 | 91% |
| Qwen3.5-27B Dense (36L) | 6.534 | 6.423 | 6.273 | 42% |

Validated at 512 and 8K context. NIAH retrieval passed. Benefit scales with model depth (91% on 64-layer MoE). Independently validated by @Corianas_ on NanoGPT.

**Enabled by default.** Activate manually on older builds with `TURBO_LAYER_ADAPTIVE=7` env var.

```bash
# Boundary V — boundary layers q8_0-V, rest turbo2-V
llama-server -m model.gguf -ctk q8_0 -ctv turbo2 -fa 1
```

See [full paper](papers/layer-aware-v-compression.md).

## Prefill Context Scaling (Verified 2K-32K)

| Context | turbo4 tok/s | turbo3 tok/s | q8_0 tok/s | turbo4/q8_0 | turbo3/q8_0 |
|---------|-------------|-------------|-----------|------------|------------|
| 2K | 2682 | 2708 | 2665 | 1.01x | 1.02x |
| 4K | 2370 | 2289 | 2255 | 1.05x | 1.01x |
| 8K | 2041 | 2054 | 2002 | 1.02x | 1.03x |
| 16K | 1621 | 1698 | 1605 | 1.01x | 1.06x |
| 32K | 1141 | 1204 | 1098 | 1.04x | 1.10x |

**Prefill: both turbo3 and turbo4 match or exceed q8_0 speed.** Compressed cache uses less bandwidth.

## Decode Speed — MoE (M5 Max 128GB, Qwen3.5-35B-A3B, Sparse V)

| Config | Short (tg128) | pp32768+tg128 | Short vs q8_0 |
|--------|--------------|---------------|--------------|
| q8_0 | 85.71 tok/s | 1173.91 tok/s | — |
| **turbo4** | **79.87 tok/s** | **1060.12 tok/s** | **0.93x** |
| turbo3 | 76.84 tok/s | 1141.74 tok/s | 0.90x |

turbo4 decode is faster than turbo3 due to simpler nibble packing and direct-extract dequant.

**Real-world server benchmark (70-page PDF, ~24K context):**

| Config | Prefill tok/s | Decode tok/s | Decode vs q8_0 |
|--------|-------------|-------------|---------------|
| q8_0 | 1449.9 | 68.2 | — |
| turbo4 | 1405.9 | 63.7 | 0.93x |
| turbo3 | 1417.8 | 53.3 | 0.78x |

## NIAH Retrieval (turbo4)

| Test | q8_0 | turbo4 | turbo3 + sparse V |
|------|------|--------|-------------------|
| Single needle (33 positions) | 30/33 (90.9%) | **31/33 (93.9%)** | 9/9 (3-pos) |

turbo4 beats q8_0 on retrieval (31/33 vs 30/33). Shared failure at 8K/100% is a model weakness, not quantization. See [turbo4 resurrection](papers/turbo4-resurrection.md) for the full investigation.

## Large Model Stress Tests (M5 Max 128GB)

| Model | Params | Weights | Config | PPL | vs q8_0 | Max Context | NIAH |
|-------|--------|---------|--------|-----|---------|-------------|------|
| Llama-3.1-70B | 70B | Q4_K_M | turbo4/turbo4 | 3.461 | +6.3% | 48K | 30/30 |
| Llama-3.1-70B | 70B | Q4_K_M | turbo3/turbo3 | 3.629 | +11.4% | 48K | 30/30 |
| **Command-R+ 104B** | **104B** | **Q4_K_M** | **turbo4/turbo4** | **6.312** | **+1.9%** | **128K** | **10/10** |
| **Command-R+ 104B** | **104B** | **Q4_K_M** | **turbo3/turbo3** | **6.415** | **+3.6%** | **128K** | **10/10** |

turbo3 prefill is faster than q8_0 at 32K on both models (70B: 80.8 vs 75.2 t/s, 104B: 64.5 vs 62.3 t/s). Smaller KV cache = less memory bandwidth during attention.

104B at 128K requires raising macOS GPU memory cap: `sudo sysctl iogpu.wired_limit_mb=117964` (90% of 128GB). Without this, Metal stalls at ~49K context on 70B+ models. See [Getting Started Guide](getting-started.md) for per-RAM values.

See [M5 Max stress test](papers/m5-max-stress-test.md) for the full data.

## KL Divergence vs f16

| Cache | Mean KLD | Δp RMS | Same top-p % |
|-------|----------|--------|-------------|
| q8_0 | 0.001549 | 1.23% | 98.43% |
| **turbo4** | **0.009633** | **2.71%** | **95.98%** |
| q4_0 | 0.008091 | 2.75% | 95.83% |
| turbo3 | 0.016145 | 4.09% | 94.31% |

turbo4 KLD is 40% lower than turbo3. Same top-p agreement matches q4_0.

## Decode Speed — Dense (M5 Max 128GB, Qwen3.5-27B, Sparse V)

| Test | With sparse V | Without | Delta |
|------|-------------|---------|-------|
| Short (tg128) | 16.73 | 16.61 | +0.7% |
| 8K (pp8192+tg128) | 298.27 | 294.52 | +1.3% |
| 16K (pp16384+tg128) | 316.98 | 311.24 | +1.8% |

Dense models see smaller gains (attention is <5% of decode — FFN dominates). No regressions. Safe to enable by default.

**Sparse V dequant** skips V dequantization for positions where softmax attention weight < 1e-6. At long context, most attention weights are negligible — this saves approximately half the total dequant cost. +22.8% decode at 32K vs turbo3 without sparse V, pushing the ratio from 0.76x to 0.93x of q8_0. Sparse V introduces no additional PPL degradation beyond the underlying compression (validated at 32K with 50 chunks on wikitext-103, CI ±0.021). Benefit scales with context length. This is implemented as a minimal kernel modification.

Sparse V is not TurboQuant-specific: on q8_0 KV cache it yields a +5% decode speedup with identical PPL and NIAH, confirming this is a general attention-aware optimization rather than a compression-specific trick. See the [full paper](papers/sparse-v-dequant.md).

On M2/M1 (pre-M5), the auto-detected 4-mag LUT gives an additional +38-45% decode improvement at long context, and is additive with sparse V. See [Decode Speed Hardware Analysis](decode-speed-hardware-analysis.md) for the full 14-approach experiment log, and [Context Scaling Deep Dive](context-scaling-deep-dive.md) for the M5 Max optimization journey.

## Community Hardware: CUDA (RTX 3090)

Tested by @jaker86 on RTX 3090. Model: Qwen3.5-9B Q4_K_M. Build from [signalnine's CUDA fork](https://github.com/signalnine/llama-cpp-turboquant-cuda) PR #24.

| Config | K | V | PPL (wikitext-2) | vs q8_0 | Decode t/s | Prefill t/s |
|--------|---|---|-----------------|---------|-----------|------------|
| q8_0 | q8_0 | q8_0 | 8.2018 | — | 102.69 | 3774 |
| turbo3 | turbo3 | turbo3 | 8.3124 | +1.3% | 98.68 | 3707 |
| turbo4 | turbo4 | turbo4 | 8.3012 | +1.2% | 95.87 | 3628 |
| turbo2 | turbo2 | turbo2 | 8.6639 | +5.6% | 98.05 | 3680 |
| mixed | turbo3 | turbo2 | 8.5312 | +4.0% | 97.32 | 3524 |
| mixed | turbo2 | turbo3 | 8.4356 | +2.9% | 96.61 | 3608 |

CUDA decode within 4-7% of q8_0 across all configs. Prefill within 4-7%. Mixed K/V configs working correctly after PR #24 fix (prefill was 329 t/s before fix, now 3500+).

## Community Hardware: M1 Max 64GB

Tested by @mariotomich. Model: Qwen3.5-35B-A3B Q8_0, Sparse V ON. Real prompt: 38,596 tokens (70-pages.md), llama-cli with Qwen chat template.

| KV | Prefill t/s | Decode t/s | vs q8_0 |
|----|------------|-----------|---------|
| q8_0 | 399.0 | 12.4 | — |
| turbo2 | 406.2 | 10.8 | -12.9% |
| turbo3 | 370.4 | 7.7 | -37.9% |
| **turbo4** | **365.0** | **16.6** | **+33.9%** |

**turbo4 decode beats q8_0 by +33.9% at long context on M1 Max.** At 38K tokens, KV bandwidth savings outweigh dequant cost. Sparse V amplifies the gain. turbo3 decode regression (-37.9%) is the known M1 L2 cache wall — turbo3 dequant complexity causes cache eviction on pre-M5 hardware.

**Asymmetric q8_0-K + turbo4-V (recommended for pre-M5):**

Synthetic (llama-bench):

| KV | pp512 t/s | tg128 t/s | pp65536+tg128 t/s |
|----|-----------|-----------|-------------------|
| q8_0 | 876.1 | 39.55 | 275.0 |
| q8_0-K + turbo4-V | 894.9 (+2.2%) | 38.64 (-2.3%) | 271.0 (-1.5%) |

Asymmetric avoids the turbo3 decode regression (-37.9%) on pre-M5 hardware.

KV cache memory at 262K context:

| KV | Cache MiB | Saved | Compression |
|----|-----------|-------|-------------|
| q8_0 | 2782 | — | baseline |
| turbo4 | 1422 | 1360 MiB | 1.96x |
| q8_0-K + turbo4-V | 2102 | 680 MiB | 1.32x |

PPL on real document (70-pages.md, ctx=512, 20 chunks): q8_0 16.29, turbo4 16.44 (+0.93%), turbo3 16.42 (+0.76%), turbo2 17.22 (+5.69%).

## Community Hardware: AMD RX 9070 XT (RDNA 4, gfx1201, Windows 11)

First AMD GPU validation. First attempt — no debugging, no analysis, just raw testing out of the box. Qwen2.5-7B Q4_K_M on HIP SDK 7.1. gfx1201 detected natively — no `HSA_OVERRIDE_GFX_VERSION` needed.

| K | V | PPL (wikitext-2) | vs q8_0 | Prefill t/s | Decode t/s | Status |
|---|---|-----------------|---------|-------------|-----------|--------|
| q8_0 | q8_0 | 7.794 | baseline | 589.5 | 84.7 | OK |
| **q8_0** | **turbo4** | **7.876** | **+1.0%** | **588.4** | **86.8** | **recommended** |
| q8_0 | turbo3 | NaN | catastrophic | 605.1 | 87.8 | broken (HIP-specific) |
| turbo4 | turbo4 | 401.4 | catastrophic | 556.4 | 84.0 | broken (Q4_K_M) |
| turbo3 | turbo3 | 81,277 | catastrophic | 580.3 | 86.0 | broken (Q4_K_M) |

**Key findings:**
- **q8_0-K + turbo4-V confirmed on AMD** — +1.0% PPL, no speed penalty, 25% KV memory savings
- Symmetric turbo catastrophic on Q4_K_M, consistent with Metal/CUDA results
- q8_0/turbo3 produces NaN on this model (Metal gets +2.0%) — HIP-specific, under investigation
- Speed flat across configs (~85 t/s decode, ~590 t/s prefill at pp512)
- Context scaling: 0.96-0.99x vs q8_0 at pp2048-8192

See [Windows RDNA 4 Setup Guide](windows-rdna4-setup.md) for build instructions and 9 gotchas.

## Speed Optimization Journey

| Optimization | Prefill tok/s | vs q8_0 |
|-------------|--------------|---------|
| turbo3 fp32 WHT (initial) | 739 | 0.27x |
| + fp16 WHT | 1074 | 0.40x |
| + half4 vectorized butterfly | 1411 | 0.52x |
| + graph-side WHT rotation | 2095 | 0.78x |
| + block-32 storage | 2747 | 1.02x |
| **+ optimized dequant** | **2524** | **0.98x** |

> The final number (2524 at 4K) is lower than the peak (2747 at 512) because longer context is naturally slower. The key metric is the **ratio** vs q8_0, which stays flat at 0.99x. See [Speed Experiments](speed-experiments.md) for the full journey.

## Compression Quality (Python Prototype)

| Config | Compression | Cosine Sim | MSE |
|--------|-------------|------------|-----|
| TurboQuant 2-bit | 7.1× | 0.79 | 0.0047 |
| TurboQuant 2.5-bit (outlier) | **4.9×** | 0.86 | 0.0029 |
| TurboQuant 3-bit | 4.9× | 0.91 | 0.0018 |
| TurboQuant 3.5-bit (outlier) | **3.8×** | 0.95 | 0.0009 |
| TurboQuant 4-bit | 3.8× | 0.96 | 0.0007 |

## Needle-In-A-Haystack (NIAH) Retrieval

Tested using [Kamradt](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) and [NVIDIA RULER](https://github.com/NVIDIA/RULER) methodology. Qwen3.5-35B-A3B on M5 Max 128GB.

**Single Needle Retrieval (with sparse V dequant):**

| Test | q8_0 | turbo3 | turbo3 + sparse V |
|------|------|--------|-------------------|
| Single needle (9 positions) | 7/9 | 7/9 | **9/9 (100%)** |

turbo3 + sparse V achieves 9/9 in this setup (vs 7/9 baseline), suggesting a potential denoising effect from removing low-weight quantization noise. Needle positions have meaningful attention weights (well above the 1e-6 threshold) and are never skipped.

Sparse V shows no measurable impact on perplexity across all tested contexts and datasets. Observed improvements in retrieval tasks (e.g., NIAH) are treated as secondary signals and may reflect reduced quantization noise rather than fundamental model quality changes.

**Single Needle — Depth (0-100%) x Context Length (pre-sparse-V):**

| Depth | 4K | 8K | 16K | 32K |
|-------|----|----|-----|-----|
| q8_0 | 5/5 | 4/5 | 4/5 | 4/5 |
| turbo3 | 5/5 | 4/5 | 5/5 | 3/5 |

**Pre-sparse-V aggregate: q8_0 85% (17/20), turbo3 80% (16/20).** No systematic degradation from compression. N=10 needles remarkably stable (9-10/10 at every depth).

**Multi-Key with 3 Distractors (RULER MK-NIAH):**

| Cache Type | 4K | 8K | 16K | 32K |
|------------|----|----|-----|-----|
| q8_0 | 1/1 | 1/1 | 1/1 | 1/1 |
| turbo3 | 1/1 | 1/1 | 1/1 | 1/1 |

**100% retrieval accuracy with distractors through 32K.** turbo3 correctly ignores distractor needles at all context depths.

## Long-Context Perplexity (Primary Quality Metric)

50-chunk wikitext-103 at 32K context (strongest validation, CI ±0.021):

| Config | PPL | vs q8_0 | Sparse V Δ |
|--------|-----|---------|------------|
| q8_0 (8-bit KV) | 7.0638 | — | — |
| q4_0 (4-bit KV) | 7.0857 | +0.31% | — |
| turbo3 WITHOUT sparse V (3.5-bit) | 7.1796 | +1.64% | — |
| turbo3 WITH sparse V (3.5-bit) | 7.1796 | +1.64% | **0.0000** |

Note: q4_0 is included as a reference baseline. No optimization was applied to q4_0 in this work. Development focused on q8_0 and turbo3 paths.

## Key Validation

Real Qwen3-1.7B KV tensor rotation Gaussianization:
```
Raw kurtosis:       900.4  → After rotation: 2.9  (Gaussian = 3.0)
Std after rotation:  0.088388
Expected (1/√d):     0.088388
Ratio:               1.000 exactly
```

