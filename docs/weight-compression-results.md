# TQ4_1S Weight Compression — Community Results

Post-training weight compression for llama.cpp. No retraining, no calibration data. One command.

**Code:** [PR #45](https://github.com/TheTom/llama-cpp-turboquant/pull/45)
**Paper:** [weight-compression-tq4.md](papers/weight-compression-tq4.md)
**Getting Started:** [getting-started.md](getting-started.md#weight-compression-tq4_1s--experimental)

## At a Glance

| Metric | Value |
|--------|-------|
| Independent testers | **10+** |
| GPUs tested | **12+** across 5 families |
| Models tested | **11+** across 6 families |
| Compression from Q8_0 | **28–42% smaller** |
| PPL impact (Qwen/Phi) | **+0.4–3.9%** |
| PPL impact (Llama Hybrid) | **+1.3–16%** (depth dependent) |
| Regressions on uncompressed models | **Zero** |
| Weight + KV stacking penalty | **None measured** |

### Models Tested

| Model | Params | Family | Config | Compressed Size | PPL Delta |
|-------|--------|--------|--------|----------------|-----------|
| Qwen2.5-1.5B | 1.5B | Qwen | Config I | 1.28G | +1.7–1.9% |
| Qwen2.5-3B | 3B | Qwen | Config I | 2.32G | +1.73% |
| Qwen2.5-7B | 7.6B | Qwen | Config I | 5.17G | +1.71% |
| Qwen3.5-27B | 26.9B | Qwen | Config I | 19.1G | +1.3–2.5% |
| Qwen3.5-35B MoE | 34.7B | Qwen | Config I | 21.6G | +1.4% |
| Qwen2.5-72B | 72.7B | Qwen | Config I | 45.8G | +3.9% |
| Phi-4 | 14.7B | Phi | Config I | 9.9G | +0.76–1.0% |
| Llama 3.1 70B | 70.6B | Llama | Hybrid | 40.2G | +16% |
| Llama 3.1 70B | 70.6B | Llama | Premium | 49.8G | +5.8% |
| Llama 3.2-3B | 3.2B | Llama | Hybrid | 2.10G | +1.9% |
| Mistral 7B v0.3 | 7.2B | Mistral | Hybrid | 4.40G | +1.28% |
| Mistral 7B v0.3 | 7.2B | Mistral | Premium | 5.46G | +0.41% |
| Gemma 4 31B | 30.7B | Gemma | Config I | 18.9G | TBD |

---

## Will This Run on My GPU?

This table shows the largest compressed model each GPU has successfully run. It's not a full model-per-GPU matrix because "will it fit" depends on your source quant, compression config, KV cache type, and context length. The [Model Results](#model-results) section below has the exact before/after sizes so you can do the math for your setup.

| GPU | VRAM | Largest Tested | Status | Decode vs Q8_0 | Tester |
|-----|------|---------------------|--------|----------------|--------|
| **RTX 5090** | 32GB | 27B | ✅ | **107%** (load-time conversion) | Community (CUDA) |
| **RTX 4090** | 24GB | 14B Config I, 27B+KV | ✅ | 63–67% (fused), 100% (load-time) | Community (CUDA) |
| **RTX 4070 Ti** | 12GB | 9B (KV only so far) | ✅ | 2.25x faster with turbo3 KV | Community (CUDA) |
| **RTX 3090** | 24GB | 7B | ✅ | 29% (fused kernel) | Community (CUDA) |
| **2x L40S** | 96GB | 27B+ | ✅ | 81% | Community (CUDA) |
| **Dual 4090** | 48GB | 27B+ | ✅ | 71% | Community (CUDA) |
| **4090 + 4060** | mixed | 27B | ✅ | 82% | Community (CUDA) |
| **M5 Max** | 128GB | 72B | ✅ | 94–102% | Internal (Metal) |
| **M4 Max** | 64GB | 27B | ✅ | 85–99% | Community (Metal) |
| **M2 Pro** | 32GB | 7B | ✅ | ~85% | Internal (Metal) |
| **M1 Max** | 64GB | 27B | ✅ | 63% | Community (Metal) |
| **AMD RX 6600** | 8GB | 1.1B (CPU fallback) | ⚠️ | GPU matmul broken (not our bug) | Community (HIP) |

**Note:** CUDA decode speed varies by kernel version. Load-time TQ4_1S→q8_0 conversion (Community contributor) gives 100% native q8_0 speed at compressed file size.

---

## Model Results

### Qwen — Config I (full support)

| Model | Source | Compressed | Reduction | PPL Delta | Decode % | Tested By |
|-------|--------|-----------|-----------|-----------|----------|-----------|
| Qwen2.5-1.5B | 1.76G | 1.28G | -27% | +1.7–1.9% | 96% (Metal), 70% (4090) | Multiple testers |
| Qwen2.5-3B | 3.37G | 2.32G | -31% | +1.73% | 67% (4090) | Community (CUDA) |
| Qwen2.5-7B | 7.54G | 5.17G | -31% | +1.71% | 64% (4090), 99% (M4 Max) | Community (CUDA), Community (Metal) |
| Qwen3.5-27B | 26.6G | 19.1G | -28% | +1.3% (Metal), +2.5% (L40S) | 99% (M5), 85% (M4), 81% (L40S), **107% (5090)** | Multiple testers |
| Qwen3.5-35B MoE | 34.4G | 21.6G | -37% | +1.4% | 102% (Metal) | Internal (Metal) |
| Qwen2.5-72B | 72.0G | 45.8G | -38% | +3.9% (8ch) | 95% (Metal) | Internal (Metal) |

### Phi — Config I (full support)

| Model | Source | Compressed | Reduction | PPL Delta | Decode % | Tested By |
|-------|--------|-----------|-----------|-----------|----------|-----------|
| Phi-4 14B | 14.5G | 9.9G | -32% | +0.76–1.0% | 254% (Metal), 67% (4090) | Multiple testers |

### Llama — Hybrid/Premium (FFN sensitive to WHT)

Do **not** use Config I on Llama. Use Hybrid (Q4_K for all FFN) or Premium (Q5_K/Q6_K for FFN).

| Model | Config | Source | Compressed | Reduction | PPL Delta | Decode % | Tested By |
|-------|--------|--------|-----------|-----------|-----------|----------|-----------|
| Llama 3.1 70B | Hybrid | 69.8G | 40.2G | -42% | +16% | 133% (Metal) | Internal (Metal) |
| Llama 3.1 70B | Premium | 69.8G | 49.8G | -29% | +5.8% | fast | Internal (Metal) |
| Llama 3.2-3B | Hybrid | 3.19G | 2.10G | -34% | +1.9% | 98% (4090) | Community (CUDA) |
| Llama 3.2-3B | Premium | 3.19G | 2.52G | -21% | +0.78% | 93% (4090) | Community (CUDA) |

**Why Llama is different:** Llama amplifies per-layer quantization error 6–8x more than Qwen/Phi in FFN layers. The effect scales with depth — 3B Llama is fine, 70B needs Premium. See [paper Section 5.7](papers/weight-compression-tq4.md) for the full investigation.

### Mistral — Hybrid/Premium (better than expected)

| Model | Config | Source | Compressed | Reduction | PPL Delta | Decode % | Tested By |
|-------|--------|--------|-----------|-----------|-----------|----------|-----------|
| Mistral 7B v0.3 | Hybrid | 7.17G | 4.40G | -39% | +1.28% | **108%** (4090) | Community (CUDA) |
| Mistral 7B v0.3 | Premium | 7.17G | 5.46G | -24% | +0.41% | 99% (4090) | Community (CUDA) |

Mistral extends LlamaModel but quality impact is much lower than Llama 70B. Premium at +0.41% is effectively free.

### Gemma — In Progress

| Model | Status | Notes |
|-------|--------|-------|
| Gemma 4 31B | ✅ Fixed | head_dim=256 asymmetric KV bug reproduced and fixed. Pull latest PR head. |

---

## Stacking: Weights + KV Cache

Weight compression and TurboQuant KV compression stack with no additional penalty.

| Setup | Confirmed By | Notes |
|-------|-------------|-------|
| Config I + turbo3 KV (35B MoE, 32K ctx) | Internal (Metal) (M5 Max) | 59% of baseline total memory, +1.4% PPL |
| Config I + turbo4 KV (27B) | Community (CUDA) (2x L40S) | No additional penalty from stacking |
| Config I + turbo4 KV (all 10 models) | Community (RTX 4090) | Stacks across every model tested |
| Config I + turbo4 KV (27B) | Community (CUDA) (dual 4090) | Confirmed independently |
| TQ4_1S + turbo3 KV (8B, 100K ctx) | Community (RTX 4090) | 5.8 GiB total, turbo3 actually faster than f16 at long ctx |

### Independent Validation — TurboQuantDC (662 tests, RTX 4090)

An [independent implementation](https://github.com/dhawalc/turboQuantDC) built from scratch in Python/PyTorch (MIT license) tested TQ4_1S weight compression + TurboQuant KV stacking and ran a 600+ configuration sweep. This is the most comprehensive independent validation of the approach.

#### TQ4_1S Weight + turbo3 KV Stacking (Llama 3.1 8B, RTX 4090)

| Weight | Context | KV Config | Decode t/s | Notes |
|--------|--------:|-----------|----------:|-------|
| TQ4_1S | 8,192 | f16 | 78.4 | |
| TQ4_1S | 8,192 | turbo3 | **86.5** | **turbo3 faster** (less VRAM pressure) |
| TQ4_1S | 48,000 | f16 | 72.9 | near f16 ceiling |
| TQ4_1S | 56,000 | f16 | **OOM** | f16 cannot allocate |
| TQ4_1S | 65,536 | turbo3 | **85.8** | turbo3 still running |
| TQ4_1S | 100,000 | turbo3 | 72.7 | |
| TQ4_1S | 112,000 | turbo3 | **OOM** | turbo3 ceiling |

turbo3 extends max context from ~48K to ~100K (2.1x) on the same GPU.

#### VRAM Budget (Llama 3.1 8B)

| Config | Weight | KV @ 32K | Total | Max Context |
|--------|--------|----------|-------|-------------|
| Q4_K_M + f16 KV | 4.58G | ~4.0G | ~8.6G | ~48K |
| TQ4_1S + f16 KV | 4.77G | ~4.0G | ~8.8G | ~48K |
| TQ4_1S + turbo3 | 4.77G | ~1.0G | **~5.8G** | **~100K** |
| TQ3_1S + turbo3 | 3.90G | ~1.0G | **~4.9G** | **~110K+ (est)** |

#### 70B on Single RTX 4090 (KV compression only, Q2_K weights)

| Context | f16 KV | turbo3 KV | Notes |
|--------:|-------:|----------:|-------|
| 2,048 | 1.94 t/s | **2.87 t/s** | turbo3 48% faster |
| 8,192 | **OOM** | **2.68 t/s** | f16 cannot allocate |
| 16,384 | **OOM** | **2.83 t/s** | turbo3 still running |

turbo3 extends 70B max context from ~4K to ~16K (4x) on a single 4090.

#### PPL (wikitext-2, Llama 3.1 8B)

| Weight | KV Config | PPL | KV Delta |
|--------|-----------|-----|----------|
| Q4_0 | f16 | 7.50 | baseline |
| Q4_0 | q8_0/turbo3 | 7.55 | **+0.67%** |
| Q4_0 | q8_0/turbo4 | 7.53 | **+0.36%** |
| TQ3_1S | f16 | 9.46 | baseline (TQ3) |
| TQ3_1S | q8_0/turbo3 | 9.58 | +1.22% |

#### Research Findings (600+ config sweep)

| Finding | Detail |
|---------|--------|
| Boundary layer protection | First 2 + last 2 layers at higher precision recovers ~90% of quality gap. Confirmed independently. |
| QJL harmful | Paper's QJL stage hurts autoregressive quality. Random projection variance compounds across decode steps. |
| ResidualQuant | Drop-in replacement for QJL: store `sign(r_rotated)` directly. Same 1-bit budget, no random projection. Matches f16 quality. |
| Per-head bit allocation | High-entropy attention heads need +1 bit. Uniform bits waste budget on low-entropy heads. |
| FP16 hot window | Keeping last 64–128 tokens at f16 eliminates error accumulation. ~0% cost at long context (128/32K = 0.4%). |

**Note:** TQ4_1S PPL evaluation crashed on this tester's setup (`ggml_backend_tensor_copy` assert). Speed benchmarks work fine. Under investigation.

---

## Decode Speed by Hardware

| Hardware | Backend | Decode vs Q8_0 | Notes |
|----------|---------|----------------|-------|
| M5 Max 128GB | Metal | 94–102% | V2.1 fused kernel, NR0=8 |
| M4 Max 64GB | Metal | 85–99% | 99% on 7B, 85% on 27B |
| M2 Pro 32GB | Metal | ~85% | |
| M1 Max 64GB | Metal | 63% | Lower bandwidth (400 GB/s) |
| RTX 5090 | CUDA Blackwell | **107%** | Load-time conversion, Community tester |
| 2x L40S | CUDA Ada | 81% | Datacenter |
| Dual 4090 | CUDA Ada | 71% | Community (CUDA) |
| RTX 4090 | CUDA Ada | 63–67% | Pre-NR0 fused kernel |
| RTX 3090 | CUDA Ampere | 29% | Fused kernel only |

**Why CUDA varies:** Metal uses V2.1 fused kernel with NR0=8 (amortizes WHT rotation across 8 rows). CUDA fused kernel is newer and still being optimized. Load-time TQ4_1S→q8_0 conversion sidesteps this entirely, giving 100% native speed.

---

## Regression Checks

Zero regressions on uncompressed models across all tested hardware.

| Hardware | Baseline Delta | Tested By |
|----------|---------------|-----------|
| M5 Max | +0.04% (noise) | Internal (Metal) |
| M2 Pro | within noise | Internal (Metal) |
| 2x L40S | -0.2% pp, +0.04% tg (noise) | Community (CUDA) |
| RTX 5090 | Q2_K and Q4_0 zero impact | Community (CUDA) |
| RTX 4090 (Windows) | all standard types within noise | Community (CUDA) |
| RTX 4090 (WSL2) | Q4_0 through Q6_K all within noise | Community (CUDA) |

---

## Known Issues

| Issue | Status | Impact |
|-------|--------|--------|
| GCC 13.3 `extern` build error | ✅ Fixed (e9c54d5) | Build only |
| GCC 13/14 double `extern` in ops.cpp | Known, fix pending | Build only |
| Upstream attn_rot graph overflow (Phi-4) | ✅ Disabled by default | No user impact |
| CPU assert n>4096 | ✅ Fixed (21110eb) | CPU fallback only |
| Gemma 4 head_dim=256 crash | ✅ Fixed | Pull latest PR head |
| TQ4_1S PPL eval crash | Under investigation | Speed benchmarks work, PPL path crashes on some configs |
| HIP gfx1032 matmul abort | Upstream rocBLAS issue | Not TQ-specific |

## Common Pitfalls

| Mistake | What Happens | Fix |
|---------|-------------|-----|
| Q4_K_M source instead of Q8_0 | Model gets BIGGER | TQ4_1S (5.0 BPW) > Q4_K (4.5 BPW). Use Q8_0 source. |
| Loading compressed GGUF with standard llama.cpp | `failed to read tensor info` | Build from PR #45 branch. Standard llama.cpp doesn't know type IDs 44/45. |
| Missing `--allow-requantize` flag | `requantizing from type q8_0 is disabled` | Add `--allow-requantize` to quantize command. |
| Config I on Llama FFN | +16% PPL | Use Hybrid or Premium config for Llama family. |
| Mixing TQ and Q8_0 in same layer's attention | Garbage output | All 4 attention tensors must be same type per layer. |
| Windows CUDA runtime error | DLL not found | Add `C:\CUDA\bin\x64` to PATH. |

---

## Confidence Rating

| Area | Confidence | Basis |
|------|------------|-------|
| Runs on Metal | **High** | 4 Apple Silicon chips, 6+ models, zero failures |
| Runs on CUDA Ada | **High** | 4090, L40S, 5090, Windows + WSL2 + Linux |
| Runs on CUDA Ampere | **Medium** | 3090 tested, 3070 KV-only |
| Runs on AMD HIP | **Low** | Build works, GPU matmul broken on gfx1032 |
| Compression ratio (28–42%) | **High** | Consistent across all models and hardware |
| Quality (Qwen/Phi) | **High** | +0.4–3.9% PPL across 6 models, 5+ testers |
| Quality (Mistral) | **Medium** | 1 model tested, +0.41–1.28% |
| Quality (Llama) | **Medium** | Config-dependent, 3B fine but 70B needs Premium |
| Weight + KV stacking | **High** | 4+ independent confirmations |
| No regression on uncompressed | **High** | 6 hardware platforms, all pass |

---

## How to Contribute

Test on your hardware and post results on [PR #45](https://github.com/TheTom/llama-cpp-turboquant/pull/45).

```
Model:
Params:
Source quant: Q8_0
Hardware:
VRAM:
Setup (single/multi GPU):

Before (size, BPW):
After (size, BPW):
Compression %:
Config (Config I / Hybrid / Premium):

Speed:
  Baseline pp512:
  Baseline tg128:
  Compressed pp512:
  Compressed tg128:
  Compressed + turbo4 KV tg128:

PPL (if measured):
  Baseline:
  Compressed:
  Delta:

Issues:
Verdict (works / partial / broken):
```

**Important:** source must be Q8_0. Include both baseline and compressed runs. Label whether weight-only or weight+KV stacked. Crashes and failures are equally valuable.

---

*Last updated: 2026-04-03. Data from [PR #45](https://github.com/TheTom/llama-cpp-turboquant/pull/45) community testing.*
