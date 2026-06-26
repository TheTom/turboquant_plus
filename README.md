# TurboQuant+

> **🚀 TurboQuant KV cache compression is now in [vLLM](https://github.com/vllm-project/vllm)** ([PR #38479](https://github.com/vllm-project/vllm/pull/38479), merged April 2026): `--kv-cache-dtype turboquant_k8v4` and friends, with fused Triton store/decode kernels. The PR discussion drew on the asymmetric K/V findings from this repo. **Upstream llama.cpp has merged the core idea too**: Hadamard KV cache rotation ([#21038](https://github.com/ggml-org/llama.cpp/pull/21038), citing TurboQuant directly) with fast WHT kernels on CPU ([#22631](https://github.com/ggml-org/llama.cpp/pull/22631)), CUDA ([#23615](https://github.com/ggml-org/llama.cpp/pull/23615)), and Vulkan ([#23687](https://github.com/ggml-org/llama.cpp/pull/23687)). Rotation + the stock q4_0 cache is essentially turbo4's rotation stage; the PolarQuant codebook, norm extraction, and asymmetric policies remain here and in the fork.

> ### [Getting Started Guide](docs/getting-started.md) | [Configuration Recommendations](docs/turboquant-recommendations.md) | [Benchmarks](docs/benchmarks.md) | [Commercial Support](https://x.com/no_stp_on_snek)

Implementation of [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026) with implementation work, experiments, and follow-on findings beyond the base paper. Compresses transformer KV cache **3.8-6.4x** using PolarQuant + Walsh-Hadamard rotation, at near q8_0 prefill speed and ~0.9x decode throughput at long context. Validated end-to-end from 1.5B to **104B at 128K context on a MacBook** (turbo3, PPL 4.024, 74 GB peak memory).

This repository is the **research home**: the Python reference implementation, the validation papers, and the benchmark data. To run TurboQuant in an inference engine, pick from the table below. Pieces that prove useful and stable get upstreamed incrementally as small, reviewable patches.

## Run It Today

| Engine | Platform | Status | Notes |
|--------|----------|--------|-------|
| [vLLM](https://github.com/vllm-project/vllm) | CUDA / ROCm, datacenter | **Upstream, merged** | `--kv-cache-dtype turboquant_k8v4` and friends ([PR #38479](https://github.com/vllm-project/vllm/pull/38479)) |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | All backends | **Upstream, rotation merged** | Hadamard KV cache rotation ([#21038](https://github.com/ggml-org/llama.cpp/pull/21038)) + fast WHT kernels (CPU [#22631](https://github.com/ggml-org/llama.cpp/pull/22631), CUDA [#23615](https://github.com/ggml-org/llama.cpp/pull/23615), Vulkan [#23687](https://github.com/ggml-org/llama.cpp/pull/23687)). Rotation + q4_0 cache approximates turbo4; full PolarQuant codec is in the fork below |
| [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) | Metal, CUDA, HIP, CPU | **Production fork** | turbo2/3/4 KV cache + TQ3_1S/TQ4_1S weight formats; [prebuilt binaries](https://github.com/TheTom/llama-cpp-turboquant/releases) for Mac (Metal) and Windows (CUDA) |
| [mlx-swift-lm](https://github.com/ekryski/mlx-swift-lm/tree/alpha) | Apple Silicon, Swift | Active collaboration | Fastest Apple path: ~2.5x faster decode than Python mlx-lm, full TQ+ support including turbo4v2. 144 tok/s on Qwen3.5-35B-A3B MoE at 4K on M5 Max |
| [vllm-swift](https://github.com/TheTom/vllm-swift) | Apple Silicon, Swift | Active | OpenAI-compatible serving built on mlx-swift-lm; no Python in the inference hot path |
| [Atlas](https://github.com/Avarok-Cybersecurity/atlas) | Rust, Metal | Integrated | turbo4 KV cache append/decode kernels |
| [mlxcel](https://github.com/lablup/mlxcel) | Rust, MLX | Community port | [Turbo KV cache docs](https://github.com/lablup/mlxcel/blob/main/docs/turbo-kv-cache.md). Thanks to [@inureyes](https://github.com/inureyes) and the Lablup team for the careful port and upstream attribution |
| [MLX Python fork](https://github.com/TheTom/mlx/tree/feature/turboquant-plus) | Apple Silicon, Python | Experimental | `TurboKVCache` drop-in for mlx-lm and mlx-vlm; see [docs/mlx-port.md](docs/mlx-port.md) |
| [turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) | PyTorch | Independent implementation | From-scratch PyTorch TurboQuant for KV cache; applies this repo's layer-adaptive compression and Sparse V findings |
| [turboquant-vllm](https://github.com/varjoranta/turboquant-vllm) | CUDA, vLLM package | Community package | TQ+ KV cache for vLLM with fused CUDA kernels; builds on this repo's block_size=128 and K-dominance research |
| [Quansloth](https://github.com/PacifAIst/Quansloth) | Local server product | Community product | Air-gapped local AI server built on the TurboQuant+ stack |

## Key Findings

Three follow-on findings, independently validated by multiple researchers across different hardware and backends:

1. **V compression is free.** Compressing the value cache (even down to 2 bits) has zero measurable effect on attention quality when key precision is maintained. Confirmed on Metal (M5 Max), CUDA RTX 4090 (@sztlink), and CUDA RTX 3090 (@HyperionMS2040). See [asymmetric K/V paper](docs/papers/asymmetric-kv-compression.md).
2. **All quality degradation comes from K compression.** This is why asymmetric configs (q8_0-K + turbo-V) rescue models where symmetric fails. Validated across Qwen, Llama, Mistral, and Command-R+ families. See [M5 Max stress test](docs/papers/m5-max-stress-test.md).
3. **Boundary layers are disproportionately sensitive.** Protecting the first 2 + last 2 layers at higher precision recovers 37-91% of the quality gap. See [Boundary V paper](docs/papers/layer-aware-v-compression.md).

Additional experiments and writeups: [Sparse V dequant](docs/papers/sparse-v-dequant.md) (+22.8% decode at 32K, not TurboQuant-specific), [block size optimization](docs/papers/block-size-experiment.md) (5.12x compression), [turbo4 resurrection](docs/papers/turbo4-resurrection.md) (QJL hurts, PolarQuant works), [EDEN optimal-S response](docs/papers/eden-optimal-s-revisit.md) (rotation is first-order, scale is second-order).

## Quality at a Glance (M5 Max 128GB)

| Cache Type | Bits/val | Compression | PPL (wikitext-2, 512c) | vs q8_0 |
|------------|----------|-------------|----------------------|---------|
| f16 | 16.0 | 1.0x | 6.121 | -0.16% |
| q8_0 | 8.5 | 1.9x | 6.111 | baseline |
| **turbo4** | **4.125** | **3.9x** | **6.125** | **+0.23%** |
| q4_0 | 4.5 | 3.6x | 6.142 | +0.52% |
| turbo3 | 3.5† | 4.6x† | 6.176 | +1.06% |
| turbo2 | 2.5 | 6.4x | 6.507 | +6.48% |

turbo4 (4-bit PolarQuant) has the best quality after q8_0 — closer to q8_0 than q4_0, at better compression. turbo3 trades quality for maximum compression. turbo2 (2-bit) trades more quality for extreme compression — best used asymmetrically.

> †turbo3 at default block_size=32. At block_size=128, turbo3 achieves 3.125 bits/val and 5.12x compression with identical PPL. See [block size study](docs/papers/block-size-experiment.md).

> **Important: choosing the right config for your model.** TurboQuant quality depends on your base weight quantization. Models with Q8_0+ weights work well with symmetric turbo (e.g., `-ctk turbo3 -ctv turbo3`). Some low-bit models with Q4_K_M weights may benefit from asymmetric K/V: use `-ctk q8_0 -ctv turbo4` to keep K precision high while compressing V. K precision is the dominant quality factor because it controls attention routing via softmax. Bigger models absorb quantization stacking better (104B: +3.6% vs 70B: +11.4% for turbo3). Validate on your specific model. See **[Configuration Recommendations](docs/turboquant-recommendations.md)** for the full tested matrix.

Everything else lives in **[docs/benchmarks.md](docs/benchmarks.md)**: asymmetric K/V and Boundary V tables, prefill context scaling, MoE and dense decode speed, NIAH retrieval, KL divergence, 70B/104B stress tests, community results on RTX 3090, M1 Max, and AMD RX 9070 XT, and the speed optimization journey.

## Getting Started

### Python Reference Implementation

```bash
git clone https://github.com/TheTom/turboquant_plus.git
cd turboquant_plus
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Verify — should print "141 passed"
python3 -m pytest tests/ -v

# Quick compression demo (no model needed)
python3 benchmarks/demo.py

# Validate on real model KV tensors (downloads Qwen3-1.7B, ~4GB)
pip install transformers torch accelerate
python3 benchmarks/validate_real_model.py
```

Requires Python >= 3.10, NumPy >= 1.24, SciPy >= 1.10. `torch` / `transformers` / `accelerate` are optional (real-model validation only).

### Run Inference

The fastest path is a [prebuilt binary](https://github.com/TheTom/llama-cpp-turboquant/releases), or build from the fork — see the [llama-cpp-turboquant quick start](https://github.com/TheTom/llama-cpp-turboquant#quick-start) for build instructions, supported backends, and usage details.

```bash
# Server mode with TurboQuant KV cache
./build/bin/llama-server -m models/your-model.gguf \
  --jinja -ngl 99 -c 262144 -fa on \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  -np 1 --host 0.0.0.0 --port 8080
```

| Flag | Bits/val | Compression vs fp16 | Description |
|------|----------|--------------------:|-------------|
| `turbo3` | 3.5† | **4.6x**† | 3-bit PolarQuant + WHT rotation. Best compression, q8_0 speed. |
| `turbo4` | 4.125 | **3.9x** | 4-bit PolarQuant (16 centroids). Best quality. |
| `q8_0` | 8 | 2.0x | llama.cpp default quantized cache. |
| `q4_0` | 4 | 4.0x | llama.cpp 4-bit cache. |

For per-model configuration guidance (symmetric vs asymmetric, Boundary V, large-model memory caps), see the [Getting Started Guide](docs/getting-started.md) and [Configuration Recommendations](docs/turboquant-recommendations.md).

## Architecture

```
Input: KV cache vector x ∈ R^d (one attention head)
    │
    ├── Extract norm: γ = ||x||, x̂ = x/γ
    │
    ├── Random rotation: WHT + random sign flips
    │   coordinates ~ N(0, 1/d) after rotation
    │
    ├── Optimal scalar quantization (Lloyd-Max)
    │   turbo4: 16 centroids (4-bit), turbo3: 8 centroids (3-bit), turbo2: 4 centroids (2-bit)
    │
    └── Output: quantized indices + norm per block
        Compression: 3.9x (turbo4), 5.1x (turbo3), 7.5x (turbo2)
```

> **Note on QJL: reference only, not used in production.**
>
> The original TurboQuant paper (Zandieh et al. 2024, [arXiv 2406.03482](https://arxiv.org/abs/2406.03482)) includes a 1-bit QJL error-correction stage. The Python `qjl.py` here implements it for paper reproducibility.
>
> **Production drops QJL on both K and V.** QJL eliminates reconstruction bias but amplifies variance, which softmax turns into attention noise. Five independent groups confirmed (buun, scos-lab, Arclabs001, +2). See [turbo4-resurrection.md](docs/papers/turbo4-resurrection.md) for the full ablation and mechanism.
>
> If you're building on this repo: use `TurboQuantMSE` (V cache), or implement straight 4 to 8 bit PolarQuant on K. Only enable the `QJL` / `TurboQuant` (with QJL) classes if you are reproducing the original paper or doing K-side research below 8-bit.

<details>
<summary><b>Project Structure</b></summary>

```
turboquant/
├── rotation.py        # Walsh-Hadamard Transform + random sign flips
├── codebook.py        # Lloyd-Max optimal centroid computation
├── polar_quant.py     # PolarQuant — norm extraction + WHT rotation + scalar quantization
├── qjl.py            # QJL 1-bit quantizer (paper-faithful reference, see README §QJL). Not used in production.
├── turboquant.py      # Full TurboQuant pipeline
├── kv_cache.py        # KV cache integration layer
├── outlier.py         # Outlier channel strategy (2.5-bit, 3.5-bit)
├── lloyd_max.py       # Lloyd-Max quantizer implementation
├── utils.py           # Bit packing, memory measurement
├── isoquant.py        # IsoQuant (quaternion SO(4)) experimental comparison
└── rotorquant.py      # RotorQuant experimental comparison

tests/                 # 14 test files, 500+ tests
benchmarks/
├── demo.py                       # Quick compression demo
├── run_benchmark.py              # Server-based benchmark runner
├── benchmark_results.md          # Full benchmark report
├── benchmark_llama.sh            # llama.cpp benchmark script
├── benchmark_norm_correction.py  # Norm correction validation
├── benchmark_ppl_tq_vs_rq.py    # TurboQuant vs RotorQuant PPL comparison
├── temporal_decay_prototype.py   # Temporal decay experiment
├── test_with_llama.py            # Integration test at Qwen 3.5 dimensions
├── test_outlier_comparison.py    # Outlier strategy comparison
└── validate_real_model.py        # Real model KV tensor validation

docs/
├── benchmarks.md                 # Full benchmark and validation data
├── mlx-port.md                   # MLX framework port (Python)
├── changelog.md                  # v1 milestone history
├── turboquant-recommendations.md # Configuration guide (tested matrix)
├── windows-rdna4-setup.md        # Windows + AMD RDNA 4 build guide
├── papers/                       # Validation papers and experiment writeups
└── (25+ engineering docs, investigations, experiment logs)
```

</details>

## Roadmap

| Phase | Status | Details |
|-------|--------|---------|
| Core algorithms (NumPy) | ✅ | 500+ tests across 14 test files |
| Distortion validation | ✅ | Matches paper bounds (Table 2) |
| Real model validation | ✅ | Rotation validated on Qwen3 KV tensors (kurtosis 900→2.9) |
| llama.cpp C port | ✅ | Metal GPU inference working on M1 through M5 |
| Metal shader optimization | ✅ | **q8_0 speed parity**: prefill matches or beats q8_0 |
| CUDA backend | ✅ | Community-tested on RTX 3080 Ti/3090/4090/5090, DGX Spark Blackwell |
| HIP/AMD backend | ✅ | RX 9070 XT (RDNA 4) validated, gfx1201 native |
| Asymmetric K/V | ✅ | q8_0-K + turbo-V rescues Q4_K_M models |
| Boundary V | ✅ | Layer-aware V compression, 37-91% quality recovery |
| Sparse V | ✅ | Attention-gated dequant skip, +22.8% decode on MoE. [Upstream PR #21119](https://github.com/ggml-org/llama.cpp/pull/21119) |
| Block size optimization | ✅ | 32→128, 12% better compression, zero quality cost |
| vLLM upstream | ✅ | Merged as the TurboQuant attention backend ([PR #38479](https://github.com/vllm-project/vllm/pull/38479)) |
| llama.cpp upstream (rotation) | ✅ | Hadamard KV rotation + FWHT kernels merged upstream ([#21038](https://github.com/ggml-org/llama.cpp/pull/21038), [#22631](https://github.com/ggml-org/llama.cpp/pull/22631)) |
| Upstream coordination | 🔄 | llama.cpp PR preparation for the full codec ([#27](https://github.com/TheTom/turboquant_plus/issues/27)) |
| TurboQuant+ extensions | ⏳ | Adaptive bits, temporal decay, MoE-aware compression |
| MLX Swift port | 🔄 | Active collaboration with @ekryski on [mlx-swift-lm](https://github.com/ekryski/mlx-swift-lm/tree/alpha) — turbo4v2 working |

## Paper Reference

- **TurboQuant**: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- **PolarQuant**: [arXiv 2502.02617](https://arxiv.org/abs/2502.02617) (AISTATS 2026)
- **QJL**: [arXiv 2406.03482](https://arxiv.org/abs/2406.03482)
- **Google Research Blog**: [TurboQuant: Redefining AI Efficiency](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

## Docs

- [Benchmarks](docs/benchmarks.md) — full quality/speed/retrieval data, community hardware results
- [turbo4 Rematch (Jun 2026)](docs/turbo4-rematch-2026-06.md) — centroid-port-bug fix, 4.125 bpw, PDL + fused-MMA decode, asymmetric KLD; head-to-head vs `spiritbuun`
- [MLX Port](docs/mlx-port.md) — the experimental MLX Python port: results, quickstarts, MM-NIAH
- [Changelog](docs/changelog.md) — v1 milestone history
- [Quality Benchmarks](docs/quality-benchmarks.md) — perplexity validation, bisection log
- [Speed Investigation](docs/turbo-speed-investigation.md) — Metal gotchas, fp16 WHT results, optimization history
- [Speed Experiments](docs/speed-experiments.md) — the full 739 → 2747 tok/s optimization journey
- [Context Scaling Deep Dive](docs/context-scaling-deep-dive.md) — why turbo3 degraded at long context, how we fixed it
- [Pre-Rotate-Queries Investigation](docs/pre-rotate-queries-investigation.md) — why graph-side WHT failed initially
- [Quality + Speed Gate](scripts/turbo-quality-gate.sh) — pre-push script checking PPL AND context scaling ratio

## Contributing

Issues and PRs welcome. The main areas where help is needed:

1. **Upstream PR** — prepare llama.cpp contribution (CONTRIBUTING.md requirements)
2. **CUDA kernel optimization** — fused FA kernels, decode speed parity
3. **MLX memory recovery** — implement FP16 KV drop + compressed-only attention for memory-constrained long context
4. **Quality metrics** — multi-run statistics, additional task benchmarks (GSM8K, code gen, reasoning)
5. **Long context validation** — 64K+ testing across architectures

## Support

If you find this work useful, you can support it via [GitHub Sponsors](https://github.com/sponsors/TheTom) or BTC:

BTC: bc1qsfaaf6mkz2yxx2vavg2n0zgsf3qj25uh94t83rwuq7de67dey05sc3tgjx

**Commercial support:** For inference optimization and KV cache tuning engagements, DM [@no_stp_on_snek](https://x.com/no_stp_on_snek) on X.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Copyright 2026 Tom Turney.

Based on Google Research's TurboQuant paper (arXiv 2504.19874).
