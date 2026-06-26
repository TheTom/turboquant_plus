# Changelog / Milestones

Major milestones from the v1 development cycle. The optimization history tables are in [benchmarks.md](benchmarks.md) and [speed-experiments.md](speed-experiments.md).

## v1 Complete, Speed Optimized, Community-Tested

- 511+ Python tests, 100% code coverage on diagnostics
- C port integrated into llama.cpp with Metal GPU kernels
- `--cache-type-k turbo3 --cache-type-v turbo3` works on Apple Silicon (turbo2/turbo3/turbo4 all supported)
- **turbo2 Metal support**: 2-bit, 6.4x compression, +6.48% PPL — for extreme memory pressure or asymmetric K/V
- **q8_0 prefill speed parity achieved** (2747 vs 2694 tok/s)
- **Norm correction**: PPL beats q8_0 on CUDA (-1.17%), +1.1% on Metal (ported from @spiritbuun)
- **4-mag LUT**: auto-detected on M1/M2/M3/M4, +38-45% decode at long context
- **Layer-adaptive mode 2**: q8_0 quality at 3.5x compression (last 8 layers at q8_0)
- **Temporal decay**: 30-34% memory savings at long context (experiment branch)
- **NIAH retrieval**: 9/9 single needle with sparse V (vs 7/9 baseline), 100% multi-key through 32K. 30/30 on Llama-70B, 10/10 on Command-R+ 104B
- **14 decode approaches tested** on M2 Pro — comprehensive hardware analysis
- **Stress tested up to 104B**: Command-R+ 104B Q4_K_M at 128K context (PPL 4.024). Llama-70B Q4_K_M at 48K (PPL 4.019). turbo3 prefill faster than q8_0 at 32K on both models
- Community: 30+ testers across M1/M2/M3/M5 Mac, RTX 3080 Ti/3090/4090/5090, AMD 6800 XT/9070 XT
- Rotation Gaussianization validated on real Qwen3 KV tensors (kurtosis 900 → 2.9)
