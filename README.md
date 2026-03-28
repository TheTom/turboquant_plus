# TurboQuant+

Implementation of [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026) — KV cache compression for local LLM inference, with planned extensions beyond the paper.

> **Why "Plus"?** The base TurboQuant paper is v1. I have ideas for improvements coming post-v1 — adaptive bit allocation, temporal decay compression, expert-aware MoE compression, and more. The "plus" is what comes next.

Compresses transformer KV cache **4.6x** using PolarQuant + Walsh-Hadamard rotation. Near q8_0 prefill speed and ~0.9x decode throughput at long context (Apple Silicon).

**Key contribution:** Attention-gated KV cache decoding ("Sparse V") that skips low-weight V positions during inference. up to +22.8% decode speed at 32K context, validated on wikitext-103 (50 chunks, CI ±0.021) with no measurable PPL change. Sparse V uses attention weights as a gating signal for computation, skipping work that contributes negligibly to the output.

> **Core idea:** shift KV cache optimization from compression to attention-aware computation.

~1% perplexity increase vs q8_0 due to compression; sparse V introduces no measurable additional degradation. Sparse V ON/OFF delta = 0.000 across all tested contexts and formats.

**Not TurboQuant-specific** — Sparse V was validated across q8_0, q4_0, and turbo3 KV formats.

Validated end-to-end on Qwen 3.5 35B-A3B (MoE) on M5 Max via llama.cpp Metal.

## Status: v1 Complete, Speed Optimized, Community-Tested

- 511+ Python tests, 100% code coverage on diagnostics
- C port integrated into llama.cpp with Metal GPU kernels
- `--cache-type-k turbo3 --cache-type-v turbo3` works on Apple Silicon
- **q8_0 prefill speed parity achieved** (2747 vs 2694 tok/s)
- **Norm correction**: PPL beats q8_0 on CUDA (-1.17%), +1.1% on Metal (ported from @spiritbuun)
- **4-mag LUT**: auto-detected on M1/M2/M3/M4, +38-45% decode at long context
- **Layer-adaptive mode 2**: q8_0 quality at 3.5x compression (last 8 layers at q8_0)
- **Temporal decay**: 30-34% memory savings at long context (experiment branch)
- **NIAH retrieval**: 9/9 single needle with sparse V (vs 7/9 baseline), 100% multi-key through 32K
- **14 decode approaches tested** on M2 Pro — comprehensive hardware analysis
- Community: 10+ testers across M1/M2/M5 Mac, RTX 3090/4090/5090, AMD 6800 XT/9070
- Rotation Gaussianization validated on real Qwen3 KV tensors (kurtosis 900 → 2.9)

---

## Quality and Speed (M5 Max 128GB)

### Top-of-Tree Results

| Cache Type | Bits/val | Compression | PPL (wikitext-2, 512c) | vs q8_0 |
|------------|----------|-------------|----------------------|---------|
| f16 | 16.0 | 1.0x | 6.121 | -0.16% |
| q8_0 | 8.5 | 1.9x | 6.111 | baseline |
| **turbo4** | **4.25** | **3.8x** | **6.125** | **+0.23%** |
| q4_0 | 4.5 | 3.6x | 6.142 | +0.52% |
| turbo3 | 3.5 | 4.6x | 6.176 | +1.06% |

turbo4 (4-bit PolarQuant) has the best quality after q8_0 — closer to q8_0 than q4_0, at better compression. turbo3 trades quality for maximum compression.

### Prefill Context Scaling (Verified 2K-32K)

| Context | turbo4 tok/s | turbo3 tok/s | q8_0 tok/s | turbo4/q8_0 | turbo3/q8_0 |
|---------|-------------|-------------|-----------|------------|------------|
| 2K | 2682 | 2708 | 2665 | 1.01x | 1.02x |
| 4K | 2370 | 2289 | 2255 | 1.05x | 1.01x |
| 8K | 2041 | 2054 | 2002 | 1.02x | 1.03x |
| 16K | 1621 | 1698 | 1605 | 1.01x | 1.06x |
| 32K | 1141 | 1204 | 1098 | 1.04x | 1.10x |

**Prefill: both turbo3 and turbo4 match or exceed q8_0 speed.** Compressed cache uses less bandwidth.

### Decode Speed — MoE (M5 Max 128GB, Qwen3.5-35B-A3B, Sparse V)

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

### NIAH Retrieval (turbo4)

| Test | q8_0 | turbo4 | turbo3 + sparse V |
|------|------|--------|-------------------|
| Single needle (33 positions) | 30/33 (90.9%) | **31/33 (93.9%)** | 9/9 (3-pos) |

turbo4 beats q8_0 on retrieval (31/33 vs 30/33). Shared failure at 8K/100% is a model weakness, not quantization. See [turbo4 resurrection](docs/papers/turbo4-resurrection.md) for the full investigation.

### KL Divergence vs f16

| Cache | Mean KLD | Δp RMS | Same top-p % |
|-------|----------|--------|-------------|
| q8_0 | 0.001549 | 1.23% | 98.43% |
| **turbo4** | **0.009633** | **2.71%** | **95.98%** |
| q4_0 | 0.008091 | 2.75% | 95.83% |
| turbo3 | 0.016145 | 4.09% | 94.31% |

turbo4 KLD is 40% lower than turbo3. Same top-p agreement matches q4_0.

### Decode Speed — Dense (M5 Max 128GB, Qwen3.5-27B, Sparse V)

| Test | With sparse V | Without | Delta |
|------|-------------|---------|-------|
| Short (tg128) | 16.73 | 16.61 | +0.7% |
| 8K (pp8192+tg128) | 298.27 | 294.52 | +1.3% |
| 16K (pp16384+tg128) | 316.98 | 311.24 | +1.8% |

Dense models see smaller gains (attention is <5% of decode — FFN dominates). No regressions. Safe to enable by default.

**Sparse V dequant** skips V dequantization for positions where softmax attention weight < 1e-6. At long context, most attention weights are negligible — this saves approximately half the total dequant cost. +22.8% decode at 32K vs turbo3 without sparse V, pushing the ratio from 0.76x to 0.93x of q8_0. Sparse V introduces no additional PPL degradation beyond the underlying compression (validated at 32K with 50 chunks on wikitext-103, CI ±0.021). Benefit scales with context length. This is implemented as a minimal kernel modification.

Sparse V is not TurboQuant-specific: on q8_0 KV cache it yields a +5% decode speedup with identical PPL and NIAH, confirming this is a general attention-aware optimization rather than a compression-specific trick. See the [full paper](docs/papers/sparse-v-dequant.md).

On M2/M1 (pre-M5), the auto-detected 4-mag LUT gives an additional +38-45% decode improvement at long context, and is additive with sparse V. See [Decode Speed Hardware Analysis](docs/decode-speed-hardware-analysis.md) for the full 14-approach experiment log, and [Context Scaling Deep Dive](docs/context-scaling-deep-dive.md) for the M5 Max optimization journey.

### Speed Optimization Journey

| Optimization | Prefill tok/s | vs q8_0 |
|-------------|--------------|---------|
| turbo3 fp32 WHT (initial) | 739 | 0.27x |
| + fp16 WHT | 1074 | 0.40x |
| + half4 vectorized butterfly | 1411 | 0.52x |
| + graph-side WHT rotation | 2095 | 0.78x |
| + block-32 storage | 2747 | 1.02x |
| **+ optimized dequant** | **2524** | **0.98x** |

> The final number (2524 at 4K) is lower than the peak (2747 at 512) because longer context is naturally slower. The key metric is the **ratio** vs q8_0, which stays flat at 0.99x. See [Speed Experiments](docs/speed-experiments.md) for the full journey.

### Compression Quality (Python Prototype)

| Config | Compression | Cosine Sim | MSE |
|--------|-------------|------------|-----|
| TurboQuant 2-bit | 7.1× | 0.79 | 0.0047 |
| TurboQuant 2.5-bit (outlier) | **4.9×** | 0.86 | 0.0029 |
| TurboQuant 3-bit | 4.9× | 0.91 | 0.0018 |
| TurboQuant 3.5-bit (outlier) | **3.8×** | 0.95 | 0.0009 |
| TurboQuant 4-bit | 3.8× | 0.96 | 0.0007 |

### Needle-In-A-Haystack (NIAH) Retrieval

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

### Long-Context Perplexity (Primary Quality Metric)

50-chunk wikitext-103 at 32K context (strongest validation, CI ±0.021):

| Config | PPL | vs q8_0 | Sparse V Δ |
|--------|-----|---------|------------|
| q8_0 (8-bit KV) | 7.0638 | — | — |
| q4_0 (4-bit KV) | 7.0857 | +0.31% | — |
| turbo3 WITHOUT sparse V (3.5-bit) | 7.1796 | +1.64% | — |
| turbo3 WITH sparse V (3.5-bit) | 7.1796 | +1.64% | **0.0000** |

Note: q4_0 is included as a reference baseline. No optimization was applied to q4_0 in this work. Development focused on q8_0 and turbo3 paths.

### Key Validation

Real Qwen3-1.7B KV tensor rotation Gaussianization:
```
Raw kurtosis:       900.4  → After rotation: 2.9  (Gaussian = 3.0)
Std after rotation:  0.088388
Expected (1/√d):     0.088388
Ratio:               1.000 exactly
```

---

## Getting Started

### Prerequisites

- **Python** >= 3.10
- **NumPy** >= 1.24, **SciPy** >= 1.10
- **cmake** + C/C++ compiler (for llama.cpp build)
- **Xcode Command Line Tools** (macOS Metal build)
- **Optional**: `torch`, `transformers`, `accelerate` (~4GB download, for real model validation)

### Install the Python Prototype

```bash
git clone https://github.com/TheTom/turboquant_plus.git
cd turboquant_plus
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Verify — should print "141 passed"
python3 -m pytest tests/ -v
```

### Run the Demo

```bash
# Quick compression demo (no model needed)
python3 benchmarks/demo.py

# Validate on real model KV tensors (downloads Qwen3-1.7B, ~4GB)
pip install transformers torch accelerate
python3 benchmarks/validate_real_model.py
```

### Build llama.cpp with TurboQuant

The llama.cpp port adds two new KV cache types: `turbo3` (3.25 bits, 4.9× compression) and `turbo4` (4.25 bits, 3.8× compression).

```bash
# Clone the llama.cpp fork with TurboQuant support
git clone https://github.com/TheTom/llama-cpp-turboquant.git
cd llama-cpp-turboquant
git checkout feature/turboquant-kv-cache

# Build with Metal (Apple Silicon)
cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Build with CUDA (NVIDIA) — not yet tested
# cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
# cmake --build build -j

# Verify turbo types are available
./build/bin/llama-server --help | grep turbo
# Expected output includes: turbo3, turbo4
```

The fork modifies these files from upstream llama.cpp:
- `ggml/include/ggml.h` — new type enum entries
- `ggml/src/ggml-common.h` — block structures
- `ggml/src/ggml-quants.h` — function declarations
- `ggml/src/ggml-turbo-quant.c` — C quantize/dequantize *(new file)*
- `ggml/src/ggml.c` — type traits registration
- `ggml/src/CMakeLists.txt` — build config
- `ggml/src/ggml-metal/ggml-metal.metal` — Metal GPU kernels
- `ggml/src/ggml-metal/ggml-metal-device.m` — Metal device validation
- `common/arg.cpp` — CLI arg parsing

### Run Inference with TurboQuant KV Cache

```bash
# Server mode (for Hermes Agent, Claude Code, OpenCode, etc.)
./build/bin/llama-server \
  -m models/your-model.gguf \
  --alias "model-turbo" \
  --jinja -ngl 99 -c 262144 -fa on \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  -np 1 --metrics --host 0.0.0.0 --port 8080

# CLI mode (quick test)
./build/bin/llama-cli \
  -m models/your-model.gguf \
  -ngl 99 -c 2048 -fa on \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  -n 100 -p "Hello world" --jinja
```

### Cache Type Reference

| Flag | Bits/val | Compression vs fp16 | Description |
|------|----------|--------------------:|-------------|
| `turbo3` | 3.5 | **4.6x** | 3-bit PolarQuant + WHT rotation. Best compression, q8_0 speed. |
| `turbo4` | 4.25 | **3.8x** | 4-bit PolarQuant (16 centroids). Best quality. |
| `q8_0` | 8 | 2.0x | llama.cpp default quantized cache. |
| `q4_0` | 4 | 4.0x | llama.cpp 4-bit cache. |

---

## Architecture

```
Input: KV cache vector x ∈ R^d (one attention head)
    │
    ├── Extract norm: γ = ||x||, x̂ = x/γ
    │
    ├── Stage 1: PolarQuant (b-1 bits)
    │   Random rotation Π → coordinates ~ N(0, 1/d)
    │   → optimal scalar quantization per coordinate
    │
    ├── Stage 2: QJL (1 bit)
    │   sign(S · residual) → unbiased inner product correction
    │
    └── Output: CompressedVector(indices, signs, norms)
        Total: b bits per coordinate
```

## Project Structure

```
turboquant/
├── rotation.py      # Random rotation matrices (dense QR + fast Walsh-Hadamard)
├── codebook.py      # Optimal centroid computation (closed-form + Lloyd's)
├── polar_quant.py   # PolarQuant (Algorithm 1) — with norm extraction
├── qjl.py           # QJL 1-bit quantizer
├── turboquant.py    # Full TurboQuant (Algorithm 2)
├── kv_cache.py      # KV cache integration layer
├── outlier.py       # Outlier channel strategy (2.5-bit, 3.5-bit)
└── utils.py         # Bit packing, memory measurement

tests/               # 141 tests, 100% coverage on core modules
benchmarks/
├── demo.py                    # Quick compression demo
├── run_benchmark.py           # Server-based benchmark runner
├── benchmark_results.md       # Full benchmark report
├── test_with_llama.py         # Integration test at Qwen 3.5 dimensions
├── test_outlier_comparison.py # Outlier strategy comparison
└── validate_real_model.py     # Real model KV tensor validation
```

## Roadmap

| Phase | Status | Details |
|-------|--------|---------|
| Core algorithms (NumPy) | ✅ | 141 tests, 100% coverage |
| Distortion validation | ✅ | Matches paper bounds (Table 2) |
| Outlier channel strategy | ✅ | 2.5-bit and 3.5-bit rates |
| Real model validation | ✅ | Rotation validated on Qwen3 KV tensors (kurtosis 900→2.9) |
| llama.cpp C port | ✅ | Metal GPU inference working on M5 Max |
| Benchmarks (v1) | ✅ | MoE + Dense, 4 cache types each |
| Quality validation | ✅ | PPL 5.460 (+0.8% of q8_0) — perplexity target met |
| Metal shader optimization | ✅ | **q8_0 speed parity**: 2747 tok/s (1.02x q8_0) via graph WHT + block-32 |
| Benchmark hardening | 🔄 | Perplexity done, NIAH + multi-run pending ([#24](https://github.com/TheTom/turboquant_plus/issues/24)) |
| Upstream coordination | 🔄 | llama.cpp PR preparation ([#27](https://github.com/TheTom/turboquant_plus/issues/27)) |
| TurboQuant+ extensions | ⏳ | Adaptive bits, temporal decay, MoE-aware compression |
| CUDA backend | ⏳ | Port Metal kernels to CUDA for NVIDIA |
| MLX port | ⏳ | Last |

## Paper Reference

- **TurboQuant**: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- **PolarQuant**: [arXiv 2502.02617](https://arxiv.org/abs/2502.02617) (AISTATS 2026)
- **QJL**: [arXiv 2406.03482](https://arxiv.org/abs/2406.03482)
- **Google Research Blog**: [TurboQuant: Redefining AI Efficiency](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

## Engineering Docs

Detailed debugging logs, gotchas, and benchmarks from the llama.cpp port:

- [Quality Benchmarks](docs/quality-benchmarks.md) — perplexity validation, bisection log, top-of-tree quality+speed table
- [Speed Investigation](docs/turbo-speed-investigation.md) — Metal gotchas, fp16 WHT results, optimization history
- [Speed Experiments](docs/speed-experiments.md) — the full 739 → 2747 tok/s optimization journey (5 experiments)
- [Context Scaling Deep Dive](docs/context-scaling-deep-dive.md) — why turbo3 degraded at long context, how we fixed it (every failed approach documented)
- [Pre-Rotate-Queries Investigation](docs/pre-rotate-queries-investigation.md) — why graph-side WHT failed initially, how we fixed it
- [Quality + Speed Gate](scripts/turbo-quality-gate.sh) — pre-push script checking PPL AND context scaling ratio (required before merge)

## Contributing

Issues and PRs welcome. The main areas where help is needed:

1. **CUDA backend** — port the Metal kernels to CUDA for NVIDIA GPU support
2. **Upstream PR** — prepare llama.cpp contribution (CONTRIBUTING.md requirements)
3. **turbo4 CUDA port** — turbo4 4-bit PolarQuant validated on Metal, needs CUDA port (see [issue #17](https://github.com/TheTom/llama-cpp-turboquant/issues/17))
4. **Quality metrics** — multi-run statistics, additional task benchmarks

## Support

If you find this work useful, you can support it via [GitHub Sponsors](https://github.com/sponsors/TheTom) or BTC:

BTC: bc1qsfaaf6mkz2yxx2vavg2n0zgsf3qj25uh94t83rwuq7de67dey05sc3tgjx

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Copyright 2026 Tom Turney.

Based on Google Research's TurboQuant paper (arXiv 2504.19874).
