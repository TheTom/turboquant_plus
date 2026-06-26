# MLX Framework Port (Experimental)

TurboQuant KV cache compression is being ported to Apple's [MLX framework](https://github.com/ml-explore/mlx) for native Python/Swift inference on Apple Silicon.

**Fork:** [TheTom/mlx `feature/turboquant-plus`](https://github.com/TheTom/mlx/tree/feature/turboquant-plus)

### Results (M5 Max)

**Qwen2.5-3B 4bit — delegated KVCache (5-run avg, 500 decode tokens, [`7ad7500`](https://github.com/TheTom/mlx/commit/7ad75002bbf87f5d1774859e94e2a76ccbd3a56d)):**

| Config | Decode tok/s | vs Baseline | PPL | PPL Delta |
|--------|-------------|-------------|-----|-----------|
| Baseline (f16 KV) | 172.6 | 100% | 1.8764 | — |
| **Sym turbo4** | **171.2** | **99.2%** | 1.9083 | +1.70% |
| **Asym (K=FP16, V=turbo4)** | **171.0** | **99.0%** | 1.8859 | +0.51% |

**Quality:** Output text indistinguishable from baseline. KL divergence < 0.001, cosine similarity > 0.989.

**35B MoE (Qwen3.5-35B-A3B 8bit):**

| Config | Prefill | Decode | vs Baseline |
|--------|---------|--------|-------------|
| Baseline | 11.4 | 95.7 | 100% |
| turbo4 fused + boundary | 132.7 | **94.2** | **96%** |


**Qwen3.5-27B Dense 8bit (16/64 KV layers):**

| Config | PPL | PPL Delta | Decode | vs Baseline |
|--------|-----|-----------|--------|-------------|
| Baseline | 1.4800 | — | 17.9 | 100% |
| turbo4 asymmetric | 1.5082 | +1.91% | 15.5 | 87% |
| turbo4 symmetric | 1.5219 | +2.83% | 15.4 | 86% |

**Quality Validation (Qwen2.5-7B 8bit, dense, all 28 layers KV):**

| Test | Symmetric turbo4 | Asymmetric (K=FP16) |
|------|-----------------|---------------------|
| **KLD** | 6.86 (broken) | **0.003** |
| **Top-1 match** | 10.5% (broken) | **98.1%** |
| **NIAH** | 0/15 FAIL | **15/15 PASS** |

> [!warning] **Symmetric turbo is catastrophic on dense models.** All K layers compressed → softmax error compounds across 28 layers. Asymmetric (K=FP16, V=turbo4) is mandatory for dense architectures. Hybrid models (Qwen3.5) with delta net layers are accidentally safe because only a fraction of layers use KV cache.

**Dense models (short context, deferred compression):**

| Model | Baseline Decode | turbo4 asym Decode | PPL Delta |
|-------|----------------|-------------------|-----------|
| Qwen2.5-7B 8bit | 64.2 | 64.1 | 0.00% |
| phi-4 8bit | 32.9 | 32.7 | 0.00% |

**M2 Pro — Qwen2.5-1.5B 8bit (dense, 28/28 KV layers, asymmetric):**

| Test | Result |
|------|--------|
| KLD | 0.004 |
| Top-1 match | 96.8% |
| NIAH | 30/30 PASS |

| Context | Baseline Decode | Turbo Asymmetric | vs Baseline |
|---------|----------------|-----------------|-------------|
| 128 | 34.8 | 35.2 | 101% |
| 4096 | 46.9 | 21.6 | 46% |

M2 Pro shows more decode regression at long context — lower memory bandwidth amplifies turbo overhead.

**M5 Max Context Scaling (Qwen2.5-7B 8bit, delegated KVCache, [`7ad7500`](https://github.com/TheTom/mlx/commit/7ad75002bbf87f5d1774859e94e2a76ccbd3a56d)):**

| Context | Baseline | Sym turbo4 | vs Baseline | Asym (K=FP16) | vs Baseline |
|---------|----------|-----------|-------------|---------------|-------------|
| 512 | 63.6 | 63.6 | **100%** | 64.0 | **101%** |
| 1K | 63.1 | 62.8 | **100%** | 62.6 | **99%** |
| 2K | 62.7 | 61.8 | **98%** | 62.2 | **99%** |
| 4K | 61.0 | 60.2 | **99%** | 61.0 | **100%** |
| 8K | 58.2 | 56.9 | **98%** | 57.7 | **99%** |
| 16K | 54.6 | 53.0 | **97%** | 53.8 | **99%** |

> Previous numbers (61-83%) were measured before the delegated KVCache optimization (`7ad7500`). Root cause was `mx.concatenate` allocating new arrays every decode step × n_layers. Fixed by delegating FP16 storage to an internal KVCache with pre-allocated buffers.

**MLX Python vs llama.cpp (Qwen2.5-7B, M5 Max):**

| Framework | Prefill (400 tok) | Decode | Memory |
|-----------|------------------|--------|--------|
| llama.cpp (Q8_0) | 387 | 20.9 | 7.5 GB |
| MLX (8bit) | 243 | **21.2** | 8.5 GB |

MLX decode matches llama.cpp. Prefill 37% slower (lazy graph vs pre-compiled).

**MLX Python vs llama.cpp (M2 Pro, Qwen2.5-7B):**

| Framework | Prefill (400 tok) | Decode |
|-----------|------------------|--------|
| llama.cpp | 387 | 20.9 |
| MLX | 243 | 21.3 |

> **Note:** Future benchmark logs should record Apple Silicon power mode (Low / Auto / High) when known, as it can materially affect throughput.

### Quick Start (MLX Python)

```python
import mlx_lm
from mlx.nn.layers.turbo_kv_cache import TurboKVCache

model, tokenizer = mlx_lm.load("mlx-community/Qwen2.5-7B-Instruct-8bit")
n_layers = len(model.model.layers)
cache = [TurboKVCache(bits=4, key_bits=4) for _ in range(n_layers)]
text = mlx_lm.generate(model, tokenizer, prompt="Hello!",
                        max_tokens=200, prompt_cache=cache, verbose=True)
```

```bash
pip install git+https://github.com/TheTom/mlx.git@feature/turboquant-plus
pip install mlx-lm
```

### How it works

`TurboKVCache` is a drop-in replacement for mlx-lm's `KVCache` that adds TurboQuant 4-bit K+V compression. Compatible with **mlx-lm** and **mlx-vlm** — no framework changes needed.

**Delegated KVCache architecture** ([`7ad7500`](https://github.com/TheTom/mlx/commit/7ad75002bbf87f5d1774859e94e2a76ccbd3a56d)): During prefill, stores raw FP16. On first decode step, compresses to packed TurboQuant storage and seeds an internal `KVCache` with decoded FP16. Subsequent decode tokens go through the native KVCache (pre-allocated buffers, zero-alloc slice-assign). Packed storage updated in background via periodic batch recompression on CPU stream.

- **97–100% baseline decode speed** across 512–16K context (Qwen2.5-7B, M5 Max)
- +0.51% PPL (asymmetric), +1.70% PPL (symmetric)
- 99% answer agreement with baseline (520 multimodal samples)
- Works with stock mlx-lm and mlx-vlm, no fork needed
- All TurboQuant+ papers applied (beta centroids, dual SRHT signs, boundary layers)

### Quick Start — mlx-vlm (multimodal)

```python
from mlx_vlm import load
from mlx_vlm.models.cache import make_prompt_cache
from mlx_lm.models.cache import KVCache
from mlx.nn.layers.turbo_kv_cache import TurboKVCacheLite, compact_turbo_cache

model, processor = load("mlx-community/gemma-4-26b-a4b-it-bf16")

# Wrap KV layers with TurboKVCacheLite
cache = make_prompt_cache(model.language_model)
kv_indices = [i for i, c in enumerate(cache) if isinstance(c, KVCache)]
for idx in kv_indices:
    cache[idx] = TurboKVCacheLite(cache[idx], bits=4, key_bits=4)

# Generate as normal — prefill stores FP16
from mlx_vlm import generate
generate(model, processor, prompt="...", max_tokens=1, prompt_cache=cache)

# Compact: compress K+V to 4-bit TurboQuant
compact_turbo_cache(cache)

# Continue generating — native SDPA at full speed
generate(model, processor, prompt="Continue.", max_tokens=200, prompt_cache=cache)
```

```bash
pip install git+https://github.com/TheTom/mlx.git@feature/turboquant-plus
pip install mlx-vlm
```

### Quick Start — mlx-lm (text)

```python
import mlx_lm
from mlx.nn.layers.turbo_kv_cache import make_turbo_cache, compact_turbo_cache

model, tokenizer = mlx_lm.load("mlx-community/Qwen2.5-7B-Instruct-8bit")
cache = make_turbo_cache(model, bits=4)
mlx_lm.generate(model, tokenizer, prompt="Hello!", max_tokens=1, prompt_cache=cache)
compact_turbo_cache(cache)
mlx_lm.generate(model, tokenizer, prompt="Continue.", max_tokens=200,
                 prompt_cache=cache, verbose=True)
```

### MM-NIAH Multimodal Benchmark (520 samples)

gemma-4-26b-a4b-it · BF16 · 4-bit TQ+ Compact · MM-NIAH (val) · M5 Max 128GB

| Bucket | BL Acc | TQ+ Acc | Agree | BL Decode | TQ+ Decode | Speedup | BL KV | TQ+ KV | KV saved |
|--------|--------|---------|-------|-----------|-----------|---------|-------|--------|----------|
| ~1K | 85% | 84% | 99% | 55.1 | 54.7 | 0.99x | 0.21G | 0.19G | 10% |
| ~3K | 81% | 79% | 99% | 55.2 | 54.1 | 0.98x | 0.27G | 0.21G | 22% |
| ~7K | 80% | 81% | 99% | 54.1 | 51.7 | 0.96x | 0.37G | 0.24G | 35% |
| ~15K | 76% | 76% | 100% | 52.0 | 47.8 | 0.92x | 0.53G | 0.28G | 47% |
| ~30K | 77% | 75% | 98% | 46.9 | 40.2 | 0.86x | 0.87G | 0.36G | 59% |
| ~60K | 75% | 76% | 99% | 42.6 | 33.7 | 0.79x | 1.30G | 0.47G | 64% |
| **Total** | **79%** | **78%** | **99%** | **51.1** | **47.2** | **0.92x** | **0.58G** | **0.29G** | **50%** |

99% answer agreement with baseline across all context lengths — zero systematic quality degradation. KV savings of 10–64% where TQ+ is active. Decode speedup scales from 0.99x at ~1K to 0.79x at ~60K (dequant-once overhead on longer prefills).

