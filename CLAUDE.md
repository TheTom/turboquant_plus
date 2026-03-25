# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# TurboQuant — KV Cache Compression via PolarQuant + QJL

Implementation of the TurboQuant algorithm from ICLR 2026 (arXiv 2504.19874). Goal: working Python/NumPy prototype → llama.cpp/MLX port for local inference on Apple Silicon.

Core thesis: KV cache is the bottleneck for long-context local LLM inference. TurboQuant compresses it 6× with zero accuracy loss. Nobody has ported it outside Google's internal JAX/H100 stack yet.

---

# Development Commands

```bash
# Install (editable)
pip3 install -e ".[dev]"

# Run all tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_turboquant.py -v

# Run benchmarks
python3 benchmarks/bench_speed.py
python3 benchmarks/bench_compression.py

# Type check (if mypy installed)
python3 -m mypy turboquant/
```

---

# Architecture

```
Input vector x ∈ R^d (e.g., one attention head's K or V vector)
    │
    ▼
┌─────────────────────┐
│   PolarQuant         │  Random rotation Π → coordinates ~ Beta(d/2, d/2)
│   (b-1 bits)         │  → optimal scalar quantization per coordinate
└──────────┬──────────┘
           │ indices + residual
           ▼
┌─────────────────────┐
│   QJL                │  1-bit sign(S·r) quantization of residual
│   (1 bit)            │  → unbiased inner product preservation
└──────────┬──────────┘
           │
           ▼
  CompressedVector(indices, signs, norm)
  Total: b bits per coordinate
```

## Key Types

- `PolarQuant` — MSE-optimized quantizer (Algorithm 1). Random rotation + scalar codebook.
- `QJL` — 1-bit Quantized Johnson-Lindenstrauss. Sign quantization of residuals.
- `TurboQuant` — Full algorithm (Algorithm 2). PolarQuant(b-1) + QJL(1) = b bits total.
- `TurboQuantMSE` — MSE-only variant (no QJL). Use for V cache.
- `KVCacheCompressor` — Integration layer. TurboQuant for K cache, TurboQuantMSE for V cache.
- `CompressedVector` — Dataclass holding indices, signs, and norms.

## File Layout

```
turboquant/
├── rotation.py      # Random rotation matrices (dense QR + fast Walsh-Hadamard)
├── codebook.py      # Optimal centroid computation (closed-form + Lloyd's)
├── polar_quant.py   # PolarQuant (Algorithm 1)
├── qjl.py           # QJL 1-bit quantizer
├── turboquant.py    # Full TurboQuant (Algorithm 2) + MSE-only variant
├── kv_cache.py      # KV cache integration layer
└── utils.py         # Bit packing, memory measurement
```

---

# Data Accuracy (MANDATORY)

**Never fabricate mathematical constants, distortion bounds, or algorithm details.** All implementations must trace directly to the paper.

Sources of truth:
- Paper: arXiv 2504.19874 (TurboQuant)
- Paper: arXiv 2502.02617 (PolarQuant)
- Paper: arXiv 2406.03482 (QJL)
- Obsidian notes: `~/Documents/obsidian/Self Study/TurboQuant - KV Cache Compression Paper.md`
- Obsidian notes: `~/Documents/obsidian/Self Study/Local LLM Agentic Coding - Field Notes.md`
- Google Research blog: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

If the math isn't in the paper, it doesn't go in the code. Period.

---

# Verification (MANDATORY)

**Never claim something works without end-to-end verification.** "No errors" is NOT verification.

- For quantization: round-trip test (quantize → dequantize), measure MSE against paper bounds
- For inner products: verify `|⟨x,y⟩ - ⟨x̃,ỹ⟩|` within paper's distortion bounds for 1000+ random pairs
- For compression ratios: compute actual byte counts, not theoretical
- For speed claims: benchmark with `time.perf_counter`, not vibes

---

# Development Workflow (MANDATORY)

Every change follows this loop. No shortcuts, no skipping steps.

```
1. Create or identify GitHub Issue
   ↓
2. Write failing tests FIRST
   ↓
3. Review tests with codex-review (skill)
   ↓
4. Write code (minimal implementation to pass tests)
   ↓
5. Review code with codex-review + @roast
   ↓
6. Apply applicable fixes that make sense (not blindly)
   ↓
7. Review the fixes
   ↓
8. Verify all tests pass: python3 -m pytest tests/ -v
   ↓
9. CI must be green before commit
   ↓
10. Commit with issue reference (e.g., "feat: PolarQuant codebook #3")
   ↓
11. Close GitHub Issue with resolution comments
```

## GitHub Issues (MANDATORY)

Every unit of work requires a GitHub Issue. No exceptions.

- Create issues frequently — one per algorithm, per feature, per bug
- Use labels: `type:algorithm`, `type:test`, `type:bug`, `type:benchmark`, `type:ci`, `type:port`
- Priority: `P0` (blocking) → `P1` (core) → `P2` (normal) → `P3` (cleanup)
- Reference issues in commits
- Repo: https://github.com/TheTom/turboquant_plus

## Commit Rules (MANDATORY)

- **No pushing code to main until it has been reviewed and tested.** Zero exceptions.
- **No bulk commits.** One issue = one commit. Each commit is small, focused, and tied to a single issue.
- **Code coverage must be >95% before pushing to main.**
- Every commit message references its GitHub Issue (e.g., `feat: PolarQuant codebook #3`).
- If you realize you need to fix something unrelated, create a new issue for it — don't bundle it.

## CI (MANDATORY)

- GitHub Actions CI runs on every push and PR
- CI must pass before any commit to main
- CI runs: `python3 -m pytest tests/ -v --cov=turboquant --cov-fail-under=95`
- If CI is red, fix it before doing anything else

## Review Standards

- **codex-review**: Run gpt-5.3-codex as systematic reviewer on staged changes
- **roast**: Independent brutal review — security, correctness, edge cases, math errors
- Reviews catch: wrong constants, off-by-one in bit packing, silent precision loss, untested edge cases
- **Self-review is not review.** External eyes mandatory before merge.
- Review tests BEFORE writing implementation code — bad tests = bad code

## Paper Reference PDF

The full paper PDF is at `~/Downloads/2504.19874v1.pdf` — reference it directly when verifying algorithm details, constants, or distortion bounds.

---

# Testing

- Tests live in `tests/`
- Use `pytest` with descriptive test names
- **Tests FIRST** — write the test, watch it fail, then implement
- Every algorithm must have:
  1. Round-trip correctness test
  2. Distortion bound verification against paper's table
  3. Edge cases: zero vector, unit vector, very large/small values
  4. Batch vs single-vector consistency
- Never skip numerical tolerance checks — use `np.allclose` with explicit `atol`/`rtol`

---

# Code Rules

- **python3 only** — never `python`
- **NumPy for core math** — no PyTorch dependency in core algorithms
- **Type hints** on all public functions
- **Docstrings** with Args/Returns on all public classes and methods
- **No magic numbers** — all constants traced to paper equations with comments
- **Deterministic seeds** — all random operations take explicit `seed` parameter
- **Vectorized operations** — no Python loops over coordinates (use NumPy broadcasting)

---

# Mathematical Constants (Paper Reference)

```python
# Optimal 1-bit centroids (PolarQuant, Theorem 3.1)
# c = ±√(2/πd)
CENTROIDS_1BIT = lambda d: [-sqrt(2/(pi*d)), sqrt(2/(pi*d))]

# Optimal 2-bit centroids (PolarQuant, Table 1)
# c = {±0.453/√d, ±1.51/√d}
CENTROIDS_2BIT = lambda d: [-1.51/sqrt(d), -0.453/sqrt(d), 0.453/sqrt(d), 1.51/sqrt(d)]

# QJL dequantization constant (Theorem 4.1)
QJL_CONST = sqrt(pi/2)

# Theoretical distortion bound factor (Theorem 5.1)
# TurboQuant is within √(3π)/2 ≈ 2.7 of information-theoretic optimum
BOUND_FACTOR = sqrt(3*pi) / 2
```

## Expected Distortion Bounds (Table 2)

| Bit-width | MSE Distortion | Inner Product Distortion |
|-----------|---------------|------------------------|
| 1 | 0.36 | 1.57/d |
| 2 | 0.117 | 0.56/d |
| 3 | 0.03 | 0.18/d |
| 4 | 0.009 | 0.047/d |

---

# Target Hardware

Primary: Apple M5 Max 128 GB (llama.cpp + Metal)
Secondary: RTX 3090 24 GB (llama.cpp + CUDA)

## Why This Matters

From our local LLM experiments:
- Dense 27B KV checkpoints: **149.6 MiB each** → 7 full re-processes at 91K tokens
- MoE KV checkpoints: **62.8 MiB each** → 1 re-process at 178K tokens
- TurboQuant at 3-bit: 149 MiB → **~25 MiB** (better than MoE uncompressed)
- This changes the dense vs MoE tradeoff for local inference entirely

---

# Phase Plan

See `PLAN.md` for detailed implementation tasks.

| Phase | Goal | Status |
|-------|------|--------|
| Phase 1 | Core algorithms (NumPy) | In Progress |
| Phase 2 | Validation & benchmarks | Not Started |
| Phase 3 | KV cache integration | Not Started |
| Phase 4 | Performance optimization | Not Started |
| Phase 5 | llama.cpp / MLX port | Not Started |
