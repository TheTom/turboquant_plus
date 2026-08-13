# Getting Started with TurboQuant+

Get up and running with TurboQuant+ KV cache compression in under 5 minutes.

## Prerequisites

- A GGUF model (download from [HuggingFace](https://huggingface.co))
- CMake 3.14+
- One of: Apple Silicon Mac, NVIDIA GPU (CUDA), AMD GPU (ROCm/HIP)

## Build

```bash
git clone https://github.com/TheTom/llama-cpp-turboquant
cd llama-cpp-turboquant
git checkout feature/turboquant-kv-cache

# Apple Silicon (Metal)
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# NVIDIA (CUDA)
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

# AMD (ROCm/HIP)
cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

# AMD (ROCm/HIP) on Windows — clang requires explicit optimization flags
cmake -S . -B build -G Ninja \
  -DGPU_TARGETS=gfx1201 \
  -DGGML_HIP=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG"
cmake --build build --config Release

# Windows (CUDA, use Developer Command Prompt or WSL2)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

### CUDA build notes for containerized/Linux setups

Two easy-to-miss details if you are building the CUDA path outside a fully provisioned host dev environment:

1. If you build inside an `nvidia/cuda:*` container, pass `--gpus all` to the build container as well, not just the runtime container. Otherwise the final link step can fail with missing `libcuda.so.1` / unresolved CUDA driver symbols.
2. If you want to use `-hf` / `--hf-repo`, keep HTTPS support enabled and make sure the OpenSSL development package is installed before configuring CMake. On Debian/Ubuntu-based systems or containers this usually means `libssl-dev` plus `pkg-config`.

Example CUDA container build:

```bash
docker run --rm --gpus all \
  -v "$PWD:/src" \
  -w /src \
  nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  bash -lc '
    apt-get update &&
    apt-get install -y --no-install-recommends \
      cmake build-essential git ca-certificates libssl-dev pkg-config &&
    cmake -S . -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_OPENSSL=ON &&
    cmake --build build --config Release -j"$(nproc)"
  '
```

Without the OpenSSL development package, `llama-server -hf ...` can fail at runtime with:

```text
HTTPS is not supported. Please rebuild with one of:
  -DLLAMA_BUILD_BORINGSSL=ON
  -DLLAMA_BUILD_LIBRESSL=ON
  -DLLAMA_OPENSSL=ON
```

## Quick Start

Run a model with TurboQuant+ KV cache compression:

```bash
# Interactive chat
./build/bin/llama-cli -m model.gguf -ctk q8_0 -ctv turbo3 -fa on -ngl 99 -c 8192

# Server mode
./build/bin/llama-server -m model.gguf -ctk q8_0 -ctv turbo3 -fa on -ngl 99 -c 8192
```

## Choosing Your Config

### Safe default (works on any model)

```bash
-ctk q8_0 -ctv turbo4 -fa on
```

Asymmetric: full precision K, compressed V. Safe on all tested models including sensitive ones like Qwen2.5.

> **Known limitation:** Models with `head_dim=64` (e.g., GPT-OSS-120B) may crash or produce degraded output with turbo V compression. The WHT kernel requires sufficient dimensionality for CLT convergence, and d=64 is at the lower boundary. If you hit issues on a head_dim=64 model, fall back to `-ctk q8_0 -ctv q8_0` (no turbo compression). The paper validated at head_dim=128 and 256.

### More compression (works on most models)

```bash
-ctk q8_0 -ctv turbo3 -fa on
```

5.12x V compression with minimal quality loss (+1-2% PPL).

### Maximum compression (validated large models only)

```bash
-ctk turbo3 -ctv turbo3 -fa on
```

Symmetric. Works great on Llama 70B, Command-R+ 104B, Mistral 24B, Qwen3.5 MoE. **Do NOT use on Qwen2.5 with Q4_K_M weights** (catastrophic PPL).

### When to use what

| Your model | Recommended config |
|---|---|
| Q8_0 weights, any size | `-ctk turbo4 -ctv turbo4` or `-ctk turbo3 -ctv turbo3` |
| Q4_K_M, unknown model | `-ctk q8_0 -ctv turbo4` (safe default) |
| Q4_K_M, Qwen2.5 | `-ctk q8_0 -ctv turbo3` (must be asymmetric) |
| Q4_K_M, Llama/Mistral/Cohere 24B+ | `-ctk turbo3 -ctv turbo3` (symmetric works) |

For full details see [Configuration Recommendations](turboquant-recommendations.md).

## Built-in Optimizations

These features are enabled by default on recent builds. No extra flags needed.

### Sparse V Dequant (Metal only)

Skips V dequantization for positions where the softmax attention weight is negligible (< 1e-6). At long context, 90%+ of positions have negligible weight, so this eliminates roughly half the total dequant cost.

- +22.8% decode throughput at 32K context on MoE models
- Zero perplexity impact (validated at 32K, 50 chunks, wikitext-103)
- Works with any KV cache type (turbo3, q8_0, q4_0)
- Opt-out: `TURBO_SPARSE_V=0`

See [Sparse V paper](papers/sparse-v-dequant.md).

### Boundary V (auto-enabled with turbo2-V)

Protects the first 2 + last 2 transformer layers with q8_0-V while compressing all remaining layers with turbo2-V. Recovers 37-91% of the quality gap between turbo2 and turbo3, with no speed penalty.

- Auto-enabled when you use `-ctv turbo2`
- Opt-out: `TURBO_LAYER_ADAPTIVE=0`
- Manually enable on older builds: `TURBO_LAYER_ADAPTIVE=7`

See [Boundary V paper](papers/layer-aware-v-compression.md).

### Maximum V Compression

```bash
# turbo2-V with Boundary V (auto-enabled)
# Best for extreme memory pressure. +5-9.5% PPL.
./build/bin/llama-server -m model.gguf -ctk q8_0 -ctv turbo2 -fa on -ngl 99
```

## Benchmarking

Help the project by sharing your numbers. Here's how to generate comparable results.

> **Important: Pick the right config before benchmarking.** Symmetric turbo (e.g., `-ctk turbo3 -ctv turbo3`) is catastrophic on some model families (Qwen2.5, Qwen3 MoE) with Q4 weight quantization. If you benchmark with the wrong config, your PPL numbers will be misleading -- you'll measure the damage from a bad config, not the actual compression quality. Start with asymmetric (`-ctk q8_0 -ctv turbo4`) unless you've validated symmetric on your specific model. See [Configuration Recommendations](turboquant-recommendations.md) for which configs are safe on which models.

### Compression reference

| Config | K bpv | V bpv | Avg bpv | KV compression vs fp16 |
|--------|-------|-------|---------|------------------------|
| fp16/fp16 | 16.0 | 16.0 | 16.0 | 1.0x (baseline) |
| q8_0/q8_0 | 8.5 | 8.5 | 8.5 | 1.88x |
| q8_0/turbo4 | 8.5 | 4.25 | 6.375 | 2.51x |
| q8_0/turbo3 | 8.5 | 3.125 | 5.8125 | 2.75x |
| q8_0/turbo2 | 8.5 | 2.125 | 5.3125 | 3.01x |
| turbo4/turbo4 | 4.25 | 4.25 | 4.25 | 3.76x |
| turbo3/turbo3 | 3.125 | 3.125 | 3.125 | 5.12x |

Formula: `compression = 16 / avg_bpv`. For asymmetric: `avg_bpv = (k_bpv + v_bpv) / 2`.

### How to measure actual KV cache size

llama-server logs the KV cache allocation at startup. Look for lines like:

```
llama_kv_cache_init: Metal KV buffer size = 1234.56 MiB
```

Compare this across configs to see real memory savings. You can also use `llama-bench` output which reports KV cache size.

### Step 1: Download test data

```bash
# Wikitext-2 raw text for perplexity testing
# Option 1: Direct download (no HF account needed)
wget https://huggingface.co/nisten/llama3-8b-instruct-32k-gguf/raw/main/wiki.test.raw

# Option 2: Via Hugging Face CLI (requires HF token for some models)
# pip install huggingface-hub
# huggingface-cli download Salesforce/wikitext wikitext-2-raw-v1/test-00000-of-00001.parquet
```

### Step 2: Run baseline (always do this first)

```bash
./build/bin/llama-perplexity \
  -m model.gguf \
  -ctk q8_0 -ctv q8_0 \
  -fa on -ngl 99 \
  -f wiki.test.raw \
  -c 512 --chunks 20
```

### Step 3: Run turbo configs

```bash
# Asymmetric (safe for any model)
./build/bin/llama-perplexity \
  -m model.gguf \
  -ctk q8_0 -ctv turbo3 \
  -fa on -ngl 99 \
  -f wiki.test.raw \
  -c 512 --chunks 20

# Symmetric (if your model supports it)
./build/bin/llama-perplexity \
  -m model.gguf \
  -ctk turbo3 -ctv turbo3 \
  -fa on -ngl 99 \
  -f wiki.test.raw \
  -c 512 --chunks 20
```

### Step 4: Speed benchmarks

```bash
# Short context
./build/bin/llama-bench -m model.gguf -ctk turbo3 -ctv turbo3 -fa 1

# Long context (compare turbo3 vs q8_0 at each length)
./build/bin/llama-bench -m model.gguf -ctk q8_0 -ctv q8_0 -fa 1 -p 8192 -r 1
./build/bin/llama-bench -m model.gguf -ctk turbo3 -ctv turbo3 -fa 1 -p 8192 -r 1
./build/bin/llama-bench -m model.gguf -ctk q8_0 -ctv q8_0 -fa 1 -p 32768 -r 1
./build/bin/llama-bench -m model.gguf -ctk turbo3 -ctv turbo3 -fa 1 -p 32768 -r 1
```

### Step 5: Share results

Post your numbers on [PR #45](https://github.com/TheTom/llama-cpp-turboquant/pull/45). This is where we are tracking all test results before merge.

Include: model name, weight quantization, GPU, VRAM, turbo config, PPL, and speed numbers.

## Weight Compression (TQ4_1S) — Experimental

> **Backend support:** Metal (Apple Silicon), CUDA (NVIDIA), and AMD HIP/ROCm (tested on RX 9070 XT, RDNA 4). The quantization step (llama-quantize) works on any platform. Windows builds supported for both MSVC (CUDA) and clang (HIP).

TQ4_1S applies WHT rotation + Lloyd-Max polar quantization to model weights (not just KV cache). This is post-training quantization -- no retraining or calibration required. Apply directly to Q8_0 GGUF models.

**Code:** Weight compression is now on [main](https://github.com/TheTom/llama-cpp-turboquant/tree/feature/turboquant-kv-cache). No separate branch needed.

See the [weight compression paper](papers/weight-compression-tq4.md) for full methodology and results.

### Quick Test (copy-paste, 5 minutes)

Clone, build, compress a known-good model, and benchmark. Paste the llama-bench output back in the [PR #45 comments](https://github.com/TheTom/llama-cpp-turboquant/pull/45).

```bash
# 1. Clone and build from PR #45
git clone https://github.com/TheTom/llama-cpp-turboquant.git
cd llama-cpp-turboquant
git checkout pr/tq4-weight-compression

# Pick your backend:
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release    # Apple Silicon
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release     # NVIDIA
cmake --build build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

# 2. Download a known-good test model (Qwen3.5-27B Q8_0, 26.6 GB)
#    Fits on 32GB+ VRAM. For 24GB cards, use Qwen2.5-7B-Instruct Q8_0 instead.
huggingface-cli download Qwen/Qwen3.5-27B-GGUF qwen3.5-27b-q8_0.gguf --local-dir models/

# 3. Compress with Config I
python3 -c "
n_layers = 64  # 64 for Qwen3.5-27B, 28 for Qwen2.5-7B
boundary = 2
for i in range(boundary, n_layers - boundary):
    for t in ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up']:
        print(f'blk.{i}.{t}.weight=tq4_1s')
    print(f'blk.{i}.ffn_down.weight=q4_k')
" > config_i.txt

./build/bin/llama-quantize --allow-requantize --tensor-type-file config_i.txt \
  models/qwen3.5-27b-q8_0.gguf models/qwen3.5-27b-config-i.gguf Q8_0

# Show size before/after
python3 -c "
import os
src_path = 'models/qwen3.5-27b-q8_0.gguf'
dst_path = 'models/qwen3.5-27b-config-i.gguf'
src = os.path.getsize(src_path) / (1024**3)
dst = os.path.getsize(dst_path) / (1024**3)
print(f'{src_path}: {src:.2f} GB  ->  {dst_path}: {dst:.2f} GB  ({(1-dst/src)*100:.0f}% smaller)')
"

# 4. Benchmark — run all 3 and paste the output in the PR
./build/bin/llama-bench -m models/qwen3.5-27b-q8_0.gguf -fa 1 -ngl 99 -p 512 -n 128
./build/bin/llama-bench -m models/qwen3.5-27b-config-i.gguf -fa 1 -ngl 99 -p 512 -n 128
./build/bin/llama-bench -m models/qwen3.5-27b-config-i.gguf -ctk q8_0 -ctv turbo4 -fa 1 -ngl 99 -p 512 -n 128
```

Expected results (Qwen3.5-27B):
- Q8_0 source: 26.6 GB
- Config I output: ~19.1 GB (28% smaller)
- Quality: +1.3% PPL
- Speed (Metal): 94-102% of Q8_0. Speed (CUDA): varies by GPU, collecting data

**What to report:** paste all 3 llama-bench outputs in [PR #45](https://github.com/TheTom/llama-cpp-turboquant/pull/45) along with your GPU model. Crashes, errors, or unexpected output are equally valuable.

### Other Models and Configs

The Quick Test above uses Config I, which works for Qwen, Phi, and most non-Llama models. Adjust `n_layers` for your model (28 for 1.5B, 64 for 27B, 80 for 70B, etc.). Source must be Q8_0 GGUF.

For **Llama-family models**, use Hybrid (Q4_K for ALL FFN) or Premium (Q5_K/Q6_K FFN):

```bash
# Llama Hybrid: TQ4_1S attention, Q4_K all FFN
python3 -c "
n_layers = 80  # Llama 3.1 70B
for i in range(2, n_layers - 2):
    for t in ['attn_q', 'attn_k', 'attn_v', 'attn_output']:
        print(f'blk.{i}.{t}.weight=tq4_1s')
    for t in ['ffn_gate', 'ffn_up', 'ffn_down']:
        print(f'blk.{i}.{t}.weight=q4_k')
" > llama_hybrid.txt

# Llama Premium: TQ4_1S attention, Q5_K/Q6_K FFN (better quality, +8G)
python3 -c "
n_layers = 80
for i in range(4, n_layers - 4):
    for t in ['attn_q', 'attn_k', 'attn_v', 'attn_output']:
        print(f'blk.{i}.{t}.weight=tq4_1s')
    for t in ['ffn_gate', 'ffn_up']:
        print(f'blk.{i}.{t}.weight=q5_k')
    print(f'blk.{i}.ffn_down.weight=q6_k')
" > llama_premium.txt

./build/bin/llama-quantize \
  --allow-requantize \
  --tensor-type-file llama_hybrid.txt \
  model-Q8_0.gguf model-hybrid.gguf Q8_0
```

Source requirement: Q8_0 GGUF. Models already at Q4_K_M have minimal compression headroom.

### Model Compatibility Matrix

For full community benchmark data across 12+ GPUs and 11+ models, see [Community Results](weight-compression-results.md).

Based on architecture analysis and tested results. Models marked with * are predictions based on code analysis of `convert_hf_to_gguf.py` and need community validation.

### Tested Models — Validated Results

| Model | Size (Q8_0) | Size (Compressed) | Config | PPL Delta | Decode | NIAH |
|-------|-------------|-------------------|--------|-----------|--------|------|
| Qwen2.5-1.5B | 1.76G | **1.28G** | Config I | +1.9% | 96% | 6/6 |
| Qwen3.5-27B | 26.6G | **19.1G** | Config I | +1.3% | 99% | 3/3 |
| Qwen3.5-35B-A3B MoE | 34.4G | **21.6G** | Config I | +1.4% | 102% | — |
| **Qwen2.5-72B** | **72.0G** | **45.8G** | **Config I** | **+3.9%** | **95%** | **3/3** |
| Phi-4 14B | 14.5G | **9.3G** | Config I | +1.0% | **254%** | 3/3 |
| Llama 3.1 70B | 69.8G | **49.8G** | Premium | +5.8% | fast | 3/3 |
| Llama 3.1 70B | 69.8G | **40.2G** | Hybrid | +16% | 133% | 3/3 |

### Predicted Models — Based on Architecture Analysis

Models marked with * need community validation. Predictions based on `convert_hf_to_gguf.py` analysis.

| Model Family | Config | Expected PPL Delta | Size Reduction | Notes |
|---|---|---|---|---|
| Phi-3* | Config I | ~+1% | ~36% | Same family as tested Phi-4 |
| Llama 2/3/3.2* | Hybrid/Premium | ~+6-20% | ~29-42% | Same family as tested Llama 3.1 |
| CodeLlama* | Hybrid/Premium | ~+6-20% | ~29-42% | Extends LlamaModel |
| Mistral* | Hybrid/Premium | ~+6-20% | ~29-42% | Extends LlamaModel, has permutation |
| Mixtral* | Hybrid/Premium | ~+6-20% | ~29-42% | MoE, extends LlamaModel |
| Granite (IBM)* | Hybrid/Premium | ~+6-20% | ~29-42% | Extends LlamaModel |
| Llama 4* | Config I | ~+1-4% | ~30-38% | `undo_permute=False`, no permutation |
| Gemma 2/3* | Config I | ~+1-4% | ~30-38% | No permutation |
| Command-R/R+* | Config I | ~+1-4% | ~30-38% | No permutation |
| DeepSeek V2/R1* | Config I | unknown | ~30% | MLA attention, less certain |
| Falcon* | Config I | ~+1-4% | ~30-38% | No permutation |
| InternLM* | Config I | ~+1-4% | ~30-38% | No permutation |
| OLMo* | Config I | ~+1-4% | ~30-38% | No permutation |
| Yi* | Config I | ~+1-4% | ~30-38% | No permutation |

**Config I:** attn+gate/up=TQ4_1S, ffn_down=Q4_K, boundary 2+2. For Qwen, Phi, and non-Llama models.
**Hybrid:** attn=TQ4_1S, ALL FFN=Q4_K, boundary 2+2. For Llama-family. Max compression, +16% PPL.
**Premium:** attn=TQ4_1S, ffn_gate/up=Q5_K, ffn_down=Q6_K, boundary 4+4. For Llama-family. Best quality, +5.8% PPL.

The key discriminator is whether the model's GGUF conversion permutes Q/K weights for RoPE (`undo_permute` in `convert_hf_to_gguf.py`). Models without permutation tend to work well with Config I. However, permutation alone does not fully explain the sensitivity difference (see paper Section 5.7). Treat predictions as directional. **Community validation on untested models is very welcome** -- post results on [PR #45](https://github.com/TheTom/llama-cpp-turboquant/pull/45).

### Benchmark a Compressed Model

```bash
# PPL baseline (Q8_0)
./build/bin/llama-perplexity -m model-Q8_0.gguf -ngl 99 -fa 1 \
  -f wiki.test.raw -c 512 --chunks 20

# PPL compressed
./build/bin/llama-perplexity -m model-config-i.gguf -ngl 99 -fa 1 \
  -f wiki.test.raw -c 512 --chunks 20

# Speed
./build/bin/llama-bench -m model-config-i.gguf -fa 1

# Combined with TurboQuant KV compression
./build/bin/llama-server -m model-config-i.gguf -ngl 99 -fa 1 \
  -ctk q8_0 -ctv turbo3
```

## Apple Silicon: Large Models at Long Context

If you're running 70B+ models at long context on Apple Silicon, macOS caps GPU memory at ~75% of RAM by default. This causes Metal to hang at ~49K context. Fix:

```bash
# Recommended: 90% of physical RAM (safe for sustained inference)
# 128GB Mac
sudo sysctl iogpu.wired_limit_mb=117964

# 96GB Mac
sudo sysctl iogpu.wired_limit_mb=88474

# 64GB Mac
sudo sysctl iogpu.wired_limit_mb=58982
```

No reboot required. Resets on reboot.

## Resources

- [Configuration Recommendations](turboquant-recommendations.md) (full config guide with all tested models)
- [M5 Max Stress Test](papers/m5-max-stress-test.md) (70B and 104B results)
- [Sparse V Dequantization](papers/sparse-v-dequant.md)
- [Boundary V](papers/layer-aware-v-compression.md)
- [Block Size Optimization](papers/block-size-experiment.md)
- [TurboQuant paper (Google Research)](https://arxiv.org/abs/2504.19874)
