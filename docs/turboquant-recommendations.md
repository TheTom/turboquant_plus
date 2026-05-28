# TurboQuant Configuration Recommendations

Practical guidance for choosing TurboQuant settings based on your model, weight quantization, and hardware. Based on validated testing across Metal (M1 through M5), CUDA (RTX 3080 Ti through Blackwell), and HIP (AMD RDNA 4).

> **Multi-backend validated:** Metal (Apple Silicon), CUDA (NVIDIA), and HIP (AMD) all produce consistent quality results. Speed characteristics vary by backend. CUDA decode is faster than q8_0 on some models (dusterbloom fused MMA FA). Metal prefill beats q8_0 at 32K+ context on 70B.

> **Sparse V:** Enabled by default on all Metal builds. Skips V dequant for negligible attention weights. No PPL impact, +22.8% decode on MoE at 32K. Opt-out: `TURBO_SPARSE_V=0`.

> **Block size 128:** Now the default. turbo3 achieves 5.12x compression (was 4.57x at block_size=32) with zero quality cost. Validated on Metal and CUDA (PR #32 fix).

## Validated Good

These configurations produce healthy PPL in current testing:

| Model class | Config | Evidence |
|-------------|--------|----------|
| Q8_0 weights, any size | `-ctk turbo4 -ctv turbo4` | phi-4 +1.7%, 35B MoE +0.2%, 27B Dense healthy |
| Q8_0 weights, any size | `-ctk turbo3 -ctv turbo3` | phi-4 +4.2%, 35B MoE +1.1% |
| Q8_0 weights, any size | `-ctk q8_0 -ctv turbo4` | phi-4 +0.3% |
| Q8_0 weights, any size | `-ctk q8_0 -ctv turbo3` | phi-4 +1.1% |
| Q4_K_M, larger models (24B+) | `-ctk turbo3 -ctv turbo3` | Mistral-24B PPL 4.99, Llama-70B +11.4%, Command-R+ 104B +3.6% |
| Q4_K_M, larger models (70B+) | `-ctk turbo4 -ctv turbo4` | Llama-70B +6.3%, Command-R+ 104B +1.9% |
| Q4_K_M, tested sensitive models | `-ctk q8_0 -ctv turbo4` | Qwen2.5-7B +1.0% (Metal, CUDA, AMD all confirmed) |
| Q4_K_M, tested sensitive models | `-ctk q8_0 -ctv turbo3` | Qwen2.5-7B +2.0% |
| Q4_K_M, 70B+ | `-ctk q8_0 -ctv turbo2` | Llama-70B +9.5%, Command-R+ 104B +7.9% (asymmetric rescue) |

## Validated Risky

These configurations produce catastrophic PPL in at least one tested model:

| Model class | Config | Evidence |
|-------------|--------|----------|
| Q4_K_M, Qwen2.5-7B | `-ctk turbo4 -ctv turbo4` | PPL 218 (vs 6.58 baseline) |
| Q4_K_M, Qwen2.5-7B | `-ctk turbo3 -ctv turbo3` | PPL 3556 |
| Q4_K_M, Qwen2.5-1.5B | `-ctk turbo3 -ctv turbo3` | PPL 8641+ on both M5 and M2 |

Note: symmetric turbo on Q4_K_M is not universally broken. Mistral-24B, Llama-70B, and Command-R+ 104B Q4_K_M all handle it fine. Model family and size both matter. Qwen2.5 is consistently sensitive; Llama, Mistral, and Cohere are tolerant. Bigger models absorb quantization stacking better (104B turbo3 = +3.6% vs 70B turbo3 = +11.4%).

## Experimental

These configurations showed promising results but have less validation depth:

| Model class | Config | Evidence |
|-------------|--------|----------|
| Q4_K_M, tested sensitive models | `-ctk q8_0 -ctv turbo2` | Qwen2.5-7B +5.1% |
| Q8_0 weights | `-ctk q8_0 -ctv turbo2` | phi-4 +3.1% |
| Q4_K_M, Qwen2.5-7B (AMD) | `-ctk q8_0 -ctv turbo3` | NaN on HIP (Metal gets +2.0%). HIP-specific, under investigation |

### Boundary V (auto-enabled for turbo2-V)

A layer-aware V compression strategy that protects the first 2 + last 2 layers with q8_0-V while compressing all remaining layers with turbo2-V. **Auto-enabled when `-ctv turbo2` is set** on recent builds. Opt-out: `TURBO_LAYER_ADAPTIVE=0`. On older builds, activate with `TURBO_LAYER_ADAPTIVE=7`.

Validated across 4 models on Metal. Consistently recovers 37-91% of the turbo2-to-turbo3 quality gap. Benefit scales with model depth.

| Model | Layers | turbo2 PPL | Boundary V PPL | turbo3 PPL | Quality recovered |
|-------|--------|-----------|---------------|-----------|-------------------|
| phi-4-Q8_0 | 40 | 4.835 | 4.784 | 4.742 | 55% |
| Qwen2.5-7B Q4_K_M | 28 | 6.911 | 6.835 | 6.707 | 37% |
| Qwen3.5-35B MoE | 64 | 5.257 | 5.148 | 5.137 | 91% |
| Qwen3.5-27B Dense | 36 | 6.534 | 6.423 | 6.273 | 42% |

Validated at 512 and 8K context. NIAH retrieval passed. No speed penalty. Independently validated by @Corianas_ on NanoGPT.

**Exploratory Findings (2026-03-31):** Extended validation on Qwen3.5-35B MoE Q8_0 with `q8_0-K + turbo2-V` (Boundary V auto-enabled) at 512c, 8K, and 32K context. PPL matched or beat `q8_0/turbo3` at every tested length. At 32K decode on M5 Max, turbo2+BV was ~2-3% faster than turbo3-V (n=2 runs). This makes `q8_0/turbo2` the best tested MoE long-context V config on this model. See [MoE V-compression frontier](papers/moe-v-compression-frontier.md).

See [Layer-Aware V Compression](papers/layer-aware-v-compression.md) for the original Boundary V writeup.

## Recommended Starting Points

| Your situation | Start with | Why |
|---------------|------------|-----|
| Q8_0+ weights | `-ctk turbo4 -ctv turbo4` | Best quality/compression balance |
| Q8_0+ weights, need more compression | `-ctk turbo3 -ctv turbo3` | +4% PPL, 5.12x compression |
| Q4_K_M, unknown model | `-ctk q8_0 -ctv turbo4` | Safe default, V still compressed |
| Q4_K_M, validated large model (24B+) | `-ctk turbo3 -ctv turbo3` | If you've confirmed PPL is healthy |
| Q4_K_M, 70B+ | `-ctk turbo4 -ctv turbo4` | +6.3% on 70B, +1.9% on 104B. Symmetric works on large Llama/Cohere |
| Maximum V compression | `-ctk q8_0 -ctv turbo2` | +5-9.5% PPL, Boundary V auto-enabled |
| MoE long-context V compression | `-ctk q8_0 -ctv turbo2` | Tested on Qwen3.5-35B MoE: 7.53x V, PPL within 1%, 32K decode ~2-3% faster than turbo3-V |

**Important framing:** Asymmetric q8_0-K + turbo-V is a **quality/robustness rescue**, not a speed optimization. You trade some decode throughput (K is uncompressed) for quality safety on sensitive models. If your model works fine with symmetric turbo, use symmetric.

**MoE exception:** On the tested Qwen3.5-35B MoE Q8_0 setup, `q8_0/turbo2` with Boundary V (auto-enabled) is the best tested long-context V config: 7.53x V compression, PPL within 1% of q8_0 at 512c/8K/32K, quality matching or exceeding `q8_0/turbo3`, and 32K decode ~2-3% faster than turbo3-V (n=2 runs, M5 Max). This is a scoped result for this tested setup. See [MoE V-compression frontier](papers/moe-v-compression-frontier.md).

## Why K Precision Matters More Than V

The attention mechanism computes `softmax(Q * K^T) * V`. K determines which tokens receive attention weight via softmax. Softmax amplifies small errors exponentially: a small shift in Q*K scores can flip which tokens dominate the output. V errors, by contrast, scale linearly through the weighted sum.

In current testing:
- q8_0-K + turbo3-V on Qwen2.5-7B Q4_K_M gives PPL 6.71 (+2.0% vs baseline)
- turbo3-K + q8_0-V on the same model gives PPL 3556 (catastrophic)

Same total bits, opposite directions, 500x quality difference. K precision is the dominant lever.

This is why asymmetric `-ctk q8_0 -ctv turbo3` can rescue models where symmetric `-ctk turbo3 -ctv turbo3` fails. You still get V cache compression while maintaining the attention routing accuracy that K requires.

## Tested Configurations

### Metal (Apple Silicon)

All results from Metal flash attention. PPL measured on wikitext-2-raw (512 context, 4 chunks) unless noted.

#### phi-4-14B (Q8_0 weights) — healthy across all configs

| K | V | M5 PPL | M2 PPL | vs q8_0 | Status |
|---|---|--------|--------|---------|--------|
| q8_0 | q8_0 | 4.690 | 4.691 | baseline | healthy |
| turbo4 | turbo4 | 4.770 | 4.787 | +1.7% / +2.0% | healthy |
| turbo3 | turbo3 | 4.886 | 4.956 | +4.2% / +5.7% | healthy |
| q8_0 | turbo4 | 4.702 | 4.693 | +0.3% | healthy |
| q8_0 | turbo3 | 4.742 | — | +1.1% | healthy |
| q8_0 | turbo2 | 4.835 | — | +3.1% | healthy |

Cross-hardware matched: M2 Pro and M5 Max produce equivalent results.

#### Qwen2.5-7B-Instruct (Q4_K_M weights) — sensitive to symmetric turbo, rescued by asymmetric K/V

| K | V | M5 PPL | M2 PPL | vs q8_0 | Status |
|---|---|--------|--------|---------|--------|
| q8_0 | q8_0 | 6.577 | 6.579 | baseline | healthy |
| q8_0 | turbo4 | 6.644 | 6.603 | +1.0% | rescued |
| q8_0 | turbo3 | 6.707 | 6.715 | +2.0% | rescued |
| q8_0 | turbo2 | 6.911 | — | +5.1% | rescued |
| turbo4 | turbo4 | 217.7 | 227.5 | catastrophic | avoid |
| turbo3 | turbo3 | 3556 | 3778 | catastrophic | avoid |

Cross-hardware matched: both machines show identical quality patterns.

#### Llama-3.1-70B-Instruct (Q4_K_M weights, M5 Max 128GB) — tolerates symmetric turbo

| K | V | PPL | vs q8_0 | Status |
|---|---|-----|---------|--------|
| q8_0 | q8_0 | 3.257 | baseline | healthy |
| q8_0 | turbo4 | 3.301 | +1.3% | healthy |
| q8_0 | turbo3 | 3.325 | +2.1% | healthy |
| q8_0 | turbo2 | 3.568 | +9.5% | healthy |
| turbo4 | turbo4 | 3.461 | +6.3% | healthy |
| turbo3 | turbo3 | 3.629 | +11.4% | usable |
| turbo2 | turbo2 | 5.161 | +58.5% | degraded |

Asymmetric rescue works: q8_0/turbo2 = +9.5% vs symmetric turbo2/turbo2 = +58.5%. 6x improvement in quality degradation.

Long context PPL (wikitext-2-raw):

| K | V | Context | PPL |
|---|---|---------|-----|
| q8_0 | q8_0 | 8K | 3.617 |
| q8_0 | turbo4 | 8K | 3.639 |
| q8_0 | turbo3 | 8K | 3.653 |
| turbo3 | turbo3 | 32K | 4.839 |
| q8_0 | q8_0 | 48K | 3.575 |
| turbo3 | turbo3 | 48K | 4.019 |

NIAH: 30/30 perfect (turbo3 = q8_0, 5 depths x 3 context lengths). See [70B stress test](papers/m5-max-stress-test.md).

#### Command-R+ 104B (Q4_K_M weights, M5 Max 128GB) — tolerates symmetric turbo, 128K validated

| K | V | PPL | vs q8_0 | Status |
|---|---|-----|---------|--------|
| q8_0 | q8_0 | 6.192 | baseline | healthy |
| q8_0 | turbo4 | 6.211 | +0.3% | healthy |
| q8_0 | turbo3 | 6.296 | +1.7% | healthy |
| q8_0 | turbo2 | 6.678 | +7.9% | healthy |
| turbo4 | turbo4 | 6.312 | +1.9% | healthy |
| turbo3 | turbo3 | 6.415 | +3.6% | healthy |
| turbo2 | turbo2 | 7.049 | +13.8% | usable |

Largest model tested. 104B tolerates symmetric turbo better than 70B (bigger models = more headroom). turbo3 prefill faster than q8_0 at 32K (64.5 vs 62.3 t/s).

Long context PPL (turbo3/turbo3, wikitext-2-raw):

| Context | PPL | Pass time |
|---------|-----|-----------|
| 48K | 3.672 | 931s |
| 64K | 4.321 | 1481s |
| 96K | 4.170 | 2966s |
| 128K | 4.024 | 4996s |

**128K full native context achieved** by raising macOS GPU memory cap from default ~75% of physical RAM. Recommended setting is 90% to avoid kernel panics under sustained load. Peak memory 74 GB of 128 GB.

```bash
# Recommended: 90% of physical RAM (safe for sustained inference)
# 128GB Mac
sudo sysctl iogpu.wired_limit_mb=117964
# 96GB Mac
sudo sysctl iogpu.wired_limit_mb=88474
# 64GB Mac
sudo sysctl iogpu.wired_limit_mb=58982
```

NIAH: 10/10 perfect at 4K and 8K (turbo3). 16K timed out due to slow decode on 104B, not retrieval failure.

See [M5 Max stress test](papers/m5-max-stress-test.md).

#### Qwen3.5-35B-A3B MoE (Q8_0 weights) — healthy

| K | V | M5 PPL | Status |
|---|---|--------|--------|
| turbo3 | turbo3 | 5.130 | healthy |
| turbo4 | turbo4 | 5.078 | healthy |

#### Qwen3.5-27B Dense (Q8_0 weights) — healthy

| K | V | M5 PPL | Status |
|---|---|--------|--------|
| turbo3 | turbo3 | 6.339 | healthy |

#### Mistral-Small-24B-Instruct (Q4_K_M weights) — healthy at this size

| K | V | M5 PPL | Status |
|---|---|--------|--------|
| turbo3 | turbo3 | 4.987 | healthy |

This shows Q4_K_M is not universally incompatible with symmetric turbo. The 24B model has enough capacity to absorb the quantization stacking that breaks the 7B model.

### CUDA (NVIDIA)

Community-validated on RTX 3080 Ti, RTX 3090, RTX 4090, RTX 5090, and DGX Spark (Blackwell sm_121).

#### seanrasch — Qwen3.5-4B Q4_K_M (RTX 3090, block_size=128)

| K | V | PPL | vs fp16 | Status |
|---|---|-----|---------|--------|
| fp16 | fp16 | 10.037 | baseline | — |
| q8_0 | turbo4 | 10.124 | +0.9% | healthy |
| q8_0 | turbo3 | 10.163 | +1.3% | healthy |
| turbo3 | turbo3 | 10.247 | +2.1% | healthy |
| q8_0 | turbo2 | 10.568 | +5.3% | healthy |

Qwen3.5 hybrid architecture handles symmetric turbo without the blowup seen on Qwen2.x.

#### dusterbloom — Decode Speed (RTX 3090, block_size=128, fused MMA FA)

| Model | Decode vs q8_0 |
|-------|---------------|
| Gemma-3-12B | +7.3% faster |
| Qwen3.5-35B MoE | +4.2% faster |
| Nemotron-9B | +3.4% faster |
| Qwen3.5-9B | +0.8% faster |

Prefill near parity (-0.3% to +2.5% at pp8192).

#### Char__Bob — Mistral-Small-24B (RTX 3090)

q8_0 OOMs at 128K on 3090. turbo3 enables 128K (18,430 MiB fits). Decode flat ~43 t/s.

#### seanrasch — NIAH (RTX 3080 Ti)

Qwen3.5-9B turbo3/turbo3: 33/33 (100%), matches f16.

### HIP (AMD)

First AMD validation. RX 9070 XT (RDNA 4, gfx1201), Windows 11, HIP SDK 7.1. First attempt, no tuning.

#### Qwen2.5-7B-Instruct (Q4_K_M weights)

| K | V | PPL | vs q8_0 | Status |
|---|---|-----|---------|--------|
| q8_0 | q8_0 | 7.794 | baseline | OK |
| q8_0 | turbo4 | 7.876 | +1.0% | recommended |
| q8_0 | turbo3 | NaN | — | HIP-specific issue |
| turbo4 | turbo4 | 401.4 | catastrophic | Q4_K_M sensitivity |
| turbo3 | turbo3 | 81,277 | catastrophic | Q4_K_M sensitivity |

Asymmetric q8_0/turbo4 confirmed on AMD. Symmetric Q4_K_M failure consistent across all three GPU vendors. q8_0/turbo3 NaN is HIP-specific (Metal gets +2.0%). No speed penalty on working configs.

See [Windows RDNA 4 Setup Guide](windows-rdna4-setup.md) for build instructions and 9 gotchas.

## Practical Guidance

### Strong base weights (Q8_0, Q6_K, or higher)

Symmetric turbo works well. Start with turbo4 for best quality, turbo3 for more compression:

```bash
# Best quality with compression
llama-server -m model-Q8_0.gguf -ctk turbo4 -ctv turbo4 -fa 1

# More compression, still healthy
llama-server -m model-Q8_0.gguf -ctk turbo3 -ctv turbo3 -fa 1
```

### Low-bit base weights (Q4_K_M) on sensitive models

Qwen2.5-7B Q4_K_M fails catastrophically with symmetric turbo but is rescued by asymmetric K/V. Not all Q4_K_M models are sensitive. If unsure, start asymmetric:

```bash
# Recommended: near-baseline quality with V compression
llama-server -m model-Q4_K_M.gguf -ctk q8_0 -ctv turbo4 -fa 1

# More V compression, still good (+2% PPL)
llama-server -m model-Q4_K_M.gguf -ctk q8_0 -ctv turbo3 -fa 1

# Maximum V compression (+5-9.5% PPL, Boundary V auto-enabled)
llama-server -m model-Q4_K_M.gguf -ctk q8_0 -ctv turbo2 -fa 1
```

### Low-bit base weights (Q4_K_M) on larger or less sensitive models

Symmetric turbo3 works on Mistral-24B Q4_K_M (PPL 4.99), Llama-70B Q4_K_M (PPL 3.629), and Command-R+ 104B Q4_K_M (PPL 6.415). Validate on your specific model:

```bash
# Try symmetric first on large Q4_K_M models
llama-server -m large-model-Q4_K_M.gguf -ctk turbo3 -ctv turbo3 -fa 1

# Fall back to asymmetric if quality is poor
llama-server -m large-model-Q4_K_M.gguf -ctk q8_0 -ctv turbo3 -fa 1
```

### Unknown models

Start conservative and validate:

```bash
# Safe starting point for any model
llama-server -m model.gguf -ctk q8_0 -ctv turbo4 -fa 1
```

## Block Size

The default storage block size is 128 elements (changed from 32). turbo3 achieves 5.12x compression (vs 4.57x at block_size=32) with zero quality cost.

Validated on:
- **Metal:** 3 architectures (dense, dense Qwen, hybrid MoE), 3 context lengths (512, 8K, 32K), 2 Apple Silicon platforms (M5 Max, M2 Pro). Both symmetric and asymmetric paths.
- **CUDA:** Validated via PR #32 fix (warp-to-block mapping for block_size=128). PPL identical on RTX 3090 sm_86.
- **NIAH:** 3/3 pass at block_size=128 (phi-4, symmetric turbo3, 4K context).

On M2 Pro (Qwen2.5-1.5B, `q8_0-K + turbo3-V`), block_size=128 improved decode speed by 3-7%. This gain was not observed on M5 Max and may be specific to bandwidth-constrained hardware.

See [block size study](papers/block-size-experiment.md) for the full data.

## Notes and Caveats

- **Multi-backend:** Metal, CUDA, and HIP all validated. Asymmetric q8_0-K + turbo4-V confirmed across all three GPU vendors on Qwen2.5-7B Q4_K_M.
- **Model sensitivity varies:** Qwen2.5 is consistently sensitive to symmetric turbo on Q4_K_M. Llama, Mistral, and Qwen3.5 tolerate it. Test before deploying on new model families.
- **turbo2 as V cache:** turbo2-V with Boundary V auto-enabled gives +5-9.5% PPL depending on model. Boundary V recovers 37-91% of the quality gap.
- **PPL is measured at 512 context with 4 chunks unless noted.** Long-context PPL validated up to 128K on Command-R+ 104B.
- **104B at 128K context:** Confirmed on M5 Max 128GB with Command-R+ Q4_K_M. turbo3/turbo3 PPL 4.024. Requires raising macOS GPU memory cap: `sudo sysctl iogpu.wired_limit_mb=117964` (90% of 128GB, safe for sustained inference). Without this, Metal hangs at ~49K context on 70B+ models. Setting above 90% risks kernel panics under sustained load (community reported). `GGML_METAL_NO_RESIDENCY=1` is not needed (isolation testing confirmed). See [M5 Max stress test](papers/m5-max-stress-test.md).
- **turbo3 prefill faster than q8_0 at 32K:** Confirmed on both 70B (+7.4%) and 104B (+3.5%). Smaller KV cache reduces memory bandwidth during attention.
- **Multi-GPU with vision models:** On multi-GPU setups where mmproj (vision encoder) causes OOM on GPU0, set `MTMD_BACKEND_DEVICE=CUDA1` to move the vision model to GPU1. This was the key to unlocking 2x128K dual-slot + vision on Qwen3.5-122B across dual L40S 48GB (reported by @sjoerdmaessen). Without this, mmproj competes with model weights for GPU0 VRAM.
- **Community:** 50+ testers across M1/M2/M3/M5 Mac, RTX 3080 Ti/3090/4090/5090, L40S, DGX Spark Blackwell, AMD RX 9070 XT, Strix Halo, 7900 XTX.
