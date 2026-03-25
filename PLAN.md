# TurboQuant Implementation Plan

> Implementing TurboQuant KV cache compression from scratch based on the ICLR 2026 paper (arXiv 2504.19874).
> Goal: a working Python/NumPy prototype that can compress and decompress KV cache tensors, validate correctness against the paper's distortion bounds, and eventually integrate with llama.cpp or MLX.

## Paper Reference

- **Title:** TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
- **Authors:** Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni
- **Venue:** ICLR 2026
- **arXiv:** 2504.19874
- **Source:** https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

---

## Architecture Overview

TurboQuant is three algorithms composed together:

```
Input vector x ∈ R^d
        │
        ▼
┌─────────────────┐
│   PolarQuant     │  Random rotation → Beta-distributed coordinates
│   (b-1 bits)     │  → optimal scalar quantization per coordinate
└────────┬────────┘
         │ quantized + residual
         ▼
┌─────────────────┐
│      QJL         │  1-bit sign quantization of residual
│   (1 bit)        │  via Johnson-Lindenstrauss Transform
└────────┬────────┘
         │
         ▼
   TurboQuant output: (indices, qjl_signs, residual_norm)
   Total: b bits per coordinate
```

---

## Implementation Tasks

### Phase 1: Core Algorithms (NumPy)

- [ ] **Task 1.1 — Random Rotation Matrix Generation**
  - Generate orthogonal random rotation Π ∈ R^(d×d)
  - Use QR decomposition of random Gaussian matrix for proper Haar-distributed rotation
  - Must be deterministic given a seed (for reproducibility)
  - For large d, consider structured rotations (Hadamard + random sign flips) for O(d log d) instead of O(d²)

- [ ] **Task 1.2 — PolarQuant Codebook Construction**
  - For bit-width b, compute 2^b optimal centroids minimizing MSE
  - Paper gives closed-form for high-d:
    - b=1: centroids = `±√(2/πd)`
    - b=2: centroids = `{±0.453/√d, ±1.51/√d}`
  - For general b: use Lloyd's algorithm on the Beta(d/2, d/2) distribution (or Gaussian approximation for large d)
  - Store centroids as a sorted array for fast nearest-neighbor lookup

- [ ] **Task 1.3 — MSE-Optimized TurboQuant (Algorithm 1)**
  - `quantize_mse(x, Π, codebook)`:
    1. `y = Π @ x` (random rotation)
    2. `idx[j] = argmin_k |y[j] - codebook[k]|` for each coordinate (vectorized)
    3. Return idx (b-bit integers)
  - `dequantize_mse(idx, Π, codebook)`:
    1. `ỹ[j] = codebook[idx[j]]`
    2. `x̃ = Π.T @ ỹ`
    3. Return x̃

- [ ] **Task 1.4 — QJL (Quantized Johnson-Lindenstrauss)**
  - Generate random projection matrix S ∈ R^(d×d), entries ~ N(0,1)
  - `quantize_qjl(r, S)`:
    1. `qjl = sign(S @ r)` → {+1, -1}^d
    2. Return qjl (1-bit per entry)
  - `dequantize_qjl(qjl, S, gamma)`:
    1. `x̃_qjl = √(π/2) / d * gamma * S.T @ qjl`
    2. Return x̃_qjl
  - gamma = ||r||_2 (residual norm, stored as metadata)

- [ ] **Task 1.5 — Full TurboQuant (Algorithm 2 — Combines Both)**
  - `quantize(x, bit_width)`:
    1. MSE quantize at (b-1) bits → idx
    2. Compute residual: `r = x - dequantize_mse(idx)`
    3. QJL on residual: `qjl = sign(S @ r)`
    4. Store residual norm: `gamma = ||r||_2`
    5. Return (idx, qjl, gamma)
  - `dequantize(idx, qjl, gamma)`:
    1. `x̃_mse = dequantize_mse(idx)`
    2. `x̃_qjl = √(π/2) / d * gamma * S.T @ qjl`
    3. Return `x̃_mse + x̃_qjl`

### Phase 2: Validation & Benchmarks

- [ ] **Task 2.1 — Unit Tests**
  - Round-trip: quantize → dequantize, measure MSE
  - Verify MSE distortion matches paper bounds (Table from paper):
    | b | Expected MSE | Expected IP Distortion |
    |---|-------------|----------------------|
    | 1 | 0.36 | 1.57/d |
    | 2 | 0.117 | 0.56/d |
    | 3 | 0.03 | 0.18/d |
    | 4 | 0.009 | 0.047/d |
  - Verify inner product preservation: `|⟨x, y⟩ - ⟨x̃, ỹ⟩|` within bounds
  - Test with random unit vectors at d = {128, 256, 1536, 3072}

- [ ] **Task 2.2 — Quantization Speed Benchmark**
  - Paper claims ~0.002s for d=3072. Verify we're in the same ballpark.
  - Benchmark quantize/dequantize at d = {200, 1536, 3072}
  - Compare with naive product quantization

- [ ] **Task 2.3 — Compression Ratio Validation**
  - Input: fp16 KV cache tensor (16 bits/value)
  - Output at 3-bit: verify ~5.3× compression (16/3)
  - Output at 4-bit: verify ~4× compression (16/4)
  - Account for metadata overhead (rotation matrix, codebook, residual norms)

### Phase 3: KV Cache Integration

- [ ] **Task 3.1 — Batch Quantization for KV Tensors**
  - KV cache shape: (num_layers, num_heads, seq_len, head_dim)
  - Quantize along head_dim dimension (each attention head vector independently)
  - Support streaming: quantize new tokens as they arrive (online, not batch)

- [ ] **Task 3.2 — Outlier Channel Strategy (Non-Integer Bit Precision)**
  - Implement the paper's outlier split for 2.5-bit and 3.5-bit:
    - 2.5-bit: 32 outlier channels at 3 bits + 96 channels at 2 bits
    - 3.5-bit: similar split with 4-bit outliers
  - Identify outlier channels by magnitude (per-layer calibration)

- [ ] **Task 3.3 — Attention Score Computation on Compressed KV**
  - Compute `softmax(Q @ K_compressed.T / √d) @ V_compressed`
  - Use the inner product TurboQuant (Algorithm 2) for K cache
  - Use MSE TurboQuant (Algorithm 1) for V cache
  - Verify attention output matches full-precision within tolerance

### Phase 4: Performance Optimization

- [ ] **Task 4.1 — Structured Random Rotation (Fast Walsh-Hadamard)**
  - Replace dense Π with Hadamard + random sign flips
  - O(d log d) rotation instead of O(d²) matrix multiply
  - Critical for real-time KV cache compression

- [ ] **Task 4.2 — Vectorized Quantization**
  - Use NumPy broadcasting for batch quantization
  - Avoid Python loops over coordinates
  - Target: quantize 1000 vectors of d=3072 in <10ms

- [ ] **Task 4.3 — Memory-Efficient Storage**
  - Pack b-bit indices into uint8/uint16 arrays (not one int per index)
  - Pack QJL sign bits into uint8 bitfields (8 signs per byte)
  - Compute actual memory footprint vs theoretical

### Phase 5: Integration Targets (Future)

- [ ] **Task 5.1 — PyTorch Wrapper**
  - torch.nn.Module for drop-in KV cache replacement
  - Autograd-compatible (for potential fine-tuning experiments)

- [ ] **Task 5.2 — llama.cpp C Implementation**
  - Port core algorithms to C
  - Integrate with llama.cpp's KV cache management (ggml_backend)
  - Target: PR-ready patch

- [ ] **Task 5.3 — MLX Implementation**
  - Port to MLX for Apple Silicon optimization
  - Use Metal shaders for rotation + quantization

---

## Key Mathematical Constants

```python
import numpy as np

# Optimal centroids for high-dimensional case
CENTROIDS_1BIT = lambda d: np.array([-np.sqrt(2 / (np.pi * d)), np.sqrt(2 / (np.pi * d))])
CENTROIDS_2BIT = lambda d: np.array([-1.51/np.sqrt(d), -0.453/np.sqrt(d), 0.453/np.sqrt(d), 1.51/np.sqrt(d)])

# QJL dequantization constant
QJL_CONST = np.sqrt(np.pi / 2)

# Theoretical distortion bound factor
BOUND_FACTOR = np.sqrt(3 * np.pi) / 2  # ≈ 2.7
```

---

## File Structure

```
turboquant/
├── PLAN.md                    # This file
├── README.md                  # Usage instructions (after implementation)
├── turboquant/
│   ├── __init__.py
│   ├── polar_quant.py         # PolarQuant (random rotation + scalar quantization)
│   ├── qjl.py                 # QJL (1-bit Johnson-Lindenstrauss)
│   ├── turboquant.py          # Full TurboQuant (combines both)
│   ├── codebook.py            # Codebook construction (centroids)
│   ├── rotation.py            # Random rotation matrix generation
│   ├── kv_cache.py            # KV cache integration layer
│   └── utils.py               # Bit packing, memory measurement
├── tests/
│   ├── test_polar_quant.py
│   ├── test_qjl.py
│   ├── test_turboquant.py
│   ├── test_distortion.py     # Verify against paper's bounds
│   └── test_kv_cache.py
├── benchmarks/
│   ├── bench_speed.py         # Quantization speed benchmarks
│   └── bench_compression.py   # Compression ratio measurements
└── pyproject.toml
```

---

## Success Criteria

1. **Correctness:** MSE distortion within 10% of paper's reported bounds at d ≥ 128
2. **Speed:** Quantization time < 10ms for d=3072 (paper reports 0.002s)
3. **Compression:** Actual memory usage matches theoretical (b bits/coord + metadata overhead < 5%)
4. **Round-trip:** Inner product preservation error < paper's bounds for 1000 random vector pairs
