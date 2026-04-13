# Community Hardware: Apple M4 Pro 48GB

**Date**: 2026-04-13
**Hardware**: Apple M4 Pro, 48GB unified memory, macOS Darwin 25.3.0
**GPU Family**: MTLGPUFamilyApple9 (1009), Metal4 (5002)
**Build**: llama.cpp feature/turboquant-kv-cache (8590cbff9, b8814)
**Model**: Qwen2.5-1.5B-Instruct Q8_0 (1.76 GiB, 1.78B params)
**Auto-detected**: 4-mag LUT (pre-M5 hardware), sparse V dequant enabled

## Speed (llama-bench, pp512 + tg128, 3 runs)

| K | V | Prefill t/s | Decode t/s | Prefill vs q8_0 | Decode vs q8_0 |
|---|---|------------|-----------|----------------|---------------|
| q8_0 | q8_0 | 2325.95 ± 8.16 | 111.58 ± 2.33 | — | — |
| turbo4 | turbo4 | 2245.42 ± 3.03 | 75.81 ± 0.55 | 0.97x | 0.68x |
| turbo3 | turbo3 | 2242.51 ± 6.70 | 72.26 ± 1.70 | 0.96x | 0.65x |
| q8_0 | turbo4 | 2270.84 ± 9.83 | 89.77 ± 1.25 | 0.98x | 0.80x |

## Perplexity (wikitext-2, 512 ctx, 10 chunks)

| K | V | PPL | vs q8_0 |
|---|---|-----|---------|
| q8_0 | q8_0 | 11.9174 ± 0.651 | baseline |
| turbo4 | turbo4 | 6921.08 ± 521.4 | catastrophic |
| q8_0 | turbo4 | 12.0483 ± 0.659 | +1.1% |

## Python Prototype

- 551 passed, 6 skipped, 0 failed (16.39s)
- Coverage: 95% (894 statements, 43 missed)
- Core modules (codebook, kv_cache, polar_quant, qjl, rotation, turboquant, utils): 100%
- Real model validation (Qwen3-1.7B): K kurtosis 918 → post-rotation Gaussian confirmed

## Key Findings

- **Asymmetric q8_0-K + turbo4-V works on M4 Pro** — +1.1% PPL, 0.80x decode, 0.98x prefill
- **Symmetric turbo4 is catastrophic on Qwen2.5-1.5B Q8_0** — consistent with documented sensitivity on small models
- **Prefill near-parity** across all configs (96–98% of q8_0)
- **Decode regression** on M4 Pro is between M1 Max and M5 Max results, as expected for pre-M5 hardware
- **4-mag LUT auto-detected** — no manual configuration needed

## Reproduction

```bash
git clone https://github.com/TheTom/turboquant_plus.git
cd turboquant_plus
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python3 -m pytest tests/ -v  # 551 pass

# llama.cpp
git clone https://github.com/TheTom/llama-cpp-turboquant.git
cd llama-cpp-turboquant && git checkout feature/turboquant-kv-cache
cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# benchmark (download model first)
./build/bin/llama-bench -m <model.gguf> -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128 -r 3
./build/bin/llama-bench -m <model.gguf> -ngl 99 -fa 1 -ctk q8_0 -ctv turbo4 -p 512 -n 128 -r 3
```
