# TurboQuant+

Implementation of [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026) — KV cache compression for local LLM inference, with planned extensions beyond the paper.

> **Why "Plus"?** The base TurboQuant paper is v1. I have ideas for improvements coming post-v1 — adaptive bit allocation, temporal decay compression, expert-aware MoE compression, and more. The "plus" is what comes next.

Compresses transformer KV cache **up to 4.9×** using PolarQuant + QJL. Paper claims zero accuracy loss at 3.5-bit; this prototype achieves cosine similarity 0.95 at 3.5-bit on real Qwen3 KV tensors.

**Working end-to-end on Apple Silicon** — Qwen 3.5 35B-A3B MoE generating coherent text with 3-bit TurboQuant KV cache on M5 Max via llama.cpp Metal.

## Status: v1 Complete

- 141 Python tests, 100% code coverage
- C port integrated into llama.cpp with Metal GPU kernels
- `--cache-type-k turbo3 --cache-type-v turbo3` works on Apple Silicon
- Rotation Gaussianization validated on real Qwen3 KV tensors (kurtosis 900 → 2.9)

---

## Benchmark Results (M5 Max 128GB)

### Qwen 3.5 35B-A3B MoE (Q8_0)

| Cache Type | Bits/val | Prompt tok/s | Gen tok/s | KV Compression |
|------------|----------|-------------|-----------|----------------|
| q8_0 (baseline) | 8.0 | **222.8** | **85.5** | 2.0× |
| turbo4 | 4.25 | 62.6 | 10.3 | 3.8× |
| turbo3 | 3.25 | 67.3 | 10.7 | **4.9×** |

### Qwopus v2 27B Dense (Q8_0)

| Cache Type | Bits/val | Prompt tok/s | Gen tok/s | KV Compression |
|------------|----------|-------------|-----------|----------------|
| q8_0 (baseline) | 8.0 | **83.1** | **17.6** | 2.0× |
| turbo4 | 4.25 | 24.3 | 5.2 | 3.8× |
| turbo3 | 3.25 | 29.8 | 5.3 | **4.9×** |

> **Speed**: turbo3 is currently ~3-8× slower than q8_0 on generation due to WHT rotation in the dequant path. Speed ceiling test (rotation removed) shows **49.1 tok/s** (57% of q8_0) — achievable via pre-rotate-queries optimization which moves rotation from per-KV-position to per-query. Implementation in progress ([#23](https://github.com/TheTom/turboquant_plus/issues/23)).

### Compression Quality (Python Prototype)

| Config | Compression | Cosine Sim | MSE |
|--------|-------------|------------|-----|
| TurboQuant 2-bit | 7.1× | 0.79 | 0.0047 |
| TurboQuant 2.5-bit (outlier) | **4.9×** | 0.86 | 0.0029 |
| TurboQuant 3-bit | 4.9× | 0.91 | 0.0018 |
| TurboQuant 3.5-bit (outlier) | **3.8×** | 0.95 | 0.0009 |
| TurboQuant 4-bit | 3.8× | 0.96 | 0.0007 |

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
| `turbo3` | 3.25 | **4.9×** | 2-bit PolarQuant + 1-bit QJL. Best compression. |
| `turbo4` | 4.25 | **3.8×** | 3-bit PolarQuant + 1-bit QJL. Better quality. |
| `q8_0` | 8 | 2.0× | llama.cpp default quantized cache. |
| `q4_0` | 4 | 4.0× | llama.cpp 4-bit cache. |

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
| Metal shader optimization | 🔄 | Pre-rotate-queries: 10.7→49 tok/s target ([#23](https://github.com/TheTom/turboquant_plus/issues/23)) |
| Benchmark hardening | 🔄 | Perplexity, NIAH, multi-run ([#24](https://github.com/TheTom/turboquant_plus/issues/24)) |
| TurboQuant+ extensions | ⏳ | Adaptive bits, temporal decay, MoE-aware compression |
| MLX port | ⏳ | Last |

## Paper Reference

- **TurboQuant**: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- **PolarQuant**: [arXiv 2502.02617](https://arxiv.org/abs/2502.02617) (AISTATS 2026)
- **QJL**: [arXiv 2406.03482](https://arxiv.org/abs/2406.03482)
- **Google Research Blog**: [TurboQuant: Redefining AI Efficiency](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

## Contributing

Issues and PRs welcome. The main areas where help is needed:

1. **Metal shader optimization** — reduce the O(d²) per-chunk rotation to O(d log d) using Walsh-Hadamard
2. **CUDA backend** — port the Metal kernels to CUDA for NVIDIA GPU support
3. **Benchmark hardening** — perplexity evaluation, NIAH testing, multi-run statistics
4. **Quality metrics** — systematic comparison against q8_0/q4_0 on standard benchmarks

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Copyright 2026 Tom Turney.

Based on Google Research's TurboQuant paper (arXiv 2504.19874).
