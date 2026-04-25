#!/usr/bin/env python3
"""
RotorQuant vs TurboQuant Benchmark on Apple Silicon (MPS)

Tests MSE, inner product preservation, needle-in-haystack,
speed, and parameter efficiency on Mac Mini M4.

Usage: python3 benchmarks/benchmark_rotorquant.py
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant.turboquant import TurboQuant, TurboQuantMSE
from turboquant.rotorquant import RotorQuant, RotorQuantMSE

# Check for PyTorch MPS
try:
    import torch
    HAS_TORCH = True
    HAS_MPS = torch.backends.mps.is_available()
    if HAS_MPS:
        print(f"PyTorch {torch.__version__}, MPS available")
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False


def test_mse_distortion():
    print("=" * 70)
    print("TEST 1: MSE Distortion — TurboQuant vs RotorQuant")
    print("=" * 70)

    d = 128
    n = 2000
    rng = np.random.default_rng(42)

    print(f"  d={d}, n_vectors={n}\n")
    print(f"  {'bits':>4s}  {'TQ MSE':>12s}  {'RQ MSE':>12s}  {'theory':>12s}  {'winner':>8s}")
    print(f"  {'─'*4}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*8}")

    for bits in [2, 3, 4]:
        x = rng.standard_normal((n, d))
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)

        tq = TurboQuantMSE(d, bits, seed=42)
        rq = RotorQuantMSE(d, bits, seed=42)

        # TurboQuant
        idx_tq, norms_tq = tq.quantize(x)
        x_hat_tq = tq.dequantize(idx_tq, norms_tq)
        mse_tq = np.mean(np.sum((x - x_hat_tq) ** 2, axis=-1))

        # RotorQuant
        idx_rq, norms_rq = rq.quantize(x)
        x_hat_rq = rq.dequantize(idx_rq, norms_rq)
        mse_rq = np.mean(np.sum((x - x_hat_rq) ** 2, axis=-1))

        theory = np.sqrt(3) * np.pi / 2 * (1 / (4 ** bits))
        winner = "RQ" if mse_rq < mse_tq else "TQ"

        print(f"  {bits:>4d}  {mse_tq:>12.6f}  {mse_rq:>12.6f}  {theory:>12.6f}  {winner:>8s}")
    print()


def test_inner_product():
    print("=" * 70)
    print("TEST 2: Inner Product (with QJL) — TurboQuant vs RotorQuant")
    print("=" * 70)

    d = 128
    n = 2000
    rng = np.random.default_rng(42)

    print(f"  d={d}, n_pairs={n}\n")
    print(f"  {'bits':>4s}  {'':>4s}  {'bias':>10s}  {'RMSE':>10s}  {'corr':>8s}")
    print(f"  {'─'*4}  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*8}")

    for bits in [2, 3, 4]:
        x = rng.standard_normal((n, d))
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        y = rng.standard_normal((n, d))
        y = y / np.linalg.norm(y, axis=-1, keepdims=True)

        true_ip = np.sum(x * y, axis=-1)

        for label, Quant in [("TQ", TurboQuant), ("RQ", RotorQuant)]:
            q = Quant(d, bits, seed=42)
            comp = q.quantize(x)
            x_hat = q.dequantize(comp)
            est_ip = np.sum(x_hat * y, axis=-1)

            bias = np.mean(est_ip - true_ip)
            rmse = np.sqrt(np.mean((est_ip - true_ip) ** 2))
            corr = np.corrcoef(true_ip, est_ip)[0, 1]

            print(f"  {bits:>4d}  {label:>4s}  {bias:>+10.6f}  {rmse:>10.6f}  {corr:>8.4f}")
    print()


def test_needle():
    print("=" * 70)
    print("TEST 3: Needle-in-Haystack Retrieval")
    print("=" * 70)

    d = 128
    rng = np.random.default_rng(42)

    print(f"  {'bits':>4s}  {'seq':>6s}  {'TQ':>8s}  {'RQ':>8s}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*8}")

    for bits in [2, 3, 4]:
        for seq_len in [512, 2048, 8192]:
            keys = rng.standard_normal((seq_len, d))
            keys = keys / np.linalg.norm(keys, axis=-1, keepdims=True)
            needle_pos = seq_len // 3
            query = keys[needle_pos]

            results = {}
            for label, Quant in [("TQ", TurboQuant), ("RQ", RotorQuant)]:
                q = Quant(d, bits, seed=42)
                comp = q.quantize(keys)
                keys_hat = q.dequantize(comp)
                ips = keys_hat @ query
                found = np.argmax(ips) == needle_pos
                results[label] = "EXACT" if found else "MISS"

            print(f"  {bits:>4d}  {seq_len:>6d}  {results['TQ']:>8s}  {results['RQ']:>8s}")
    print()


def test_speed():
    print("=" * 70)
    print("TEST 4: Speed Benchmark (NumPy CPU)")
    print("=" * 70)

    d = 128
    bits = 3
    n_warmup = 3
    n_iter = 20
    rng = np.random.default_rng(42)

    print(f"  d={d}, bits={bits}\n")

    for n in [1000, 5000, 10000]:
        x = rng.standard_normal((n, d))
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)

        tq = TurboQuant(d, bits, seed=42)
        rq = RotorQuant(d, bits, seed=42)

        # Warmup
        for _ in range(n_warmup):
            tq.quantize(x)
            rq.quantize(x)

        # TurboQuant
        t0 = time.perf_counter()
        for _ in range(n_iter):
            tq.quantize(x)
        tq_ms = (time.perf_counter() - t0) / n_iter * 1000

        # RotorQuant
        t0 = time.perf_counter()
        for _ in range(n_iter):
            rq.quantize(x)
        rq_ms = (time.perf_counter() - t0) / n_iter * 1000

        ratio = tq_ms / rq_ms if rq_ms > 0 else float('inf')
        faster = "RQ" if ratio > 1 else "TQ"
        print(f"  n={n:>6d}: TQ={tq_ms:>8.1f} ms  RQ={rq_ms:>8.1f} ms  "
              f"({faster} {max(ratio, 1/ratio):.1f}x faster)")
    print()


def test_params():
    print("=" * 70)
    print("TEST 5: Parameter Efficiency")
    print("=" * 70)

    d = 128
    bits = 3

    tq = TurboQuant(d, bits, seed=42)
    rq = RotorQuant(d, bits, seed=42)

    # TurboQuant params: d*d rotation matrix + codebook
    tq_params = d * d + (1 << (bits - 1))  # rotation + codebook
    rq_params = rq.n_parameters

    print(f"  TurboQuant: {tq_params:,d} parameters")
    print(f"    - Rotation matrix: {d}x{d} = {d*d:,d}")
    print(f"  RotorQuant: {rq_params:,d} parameters")
    print(f"    - Rotors: {(d+2)//3} groups x 4 = {((d+2)//3)*4}")
    print(f"  Ratio: {tq_params/rq_params:.1f}x (TQ larger)")
    print()

    # Scale comparison
    print("  Scaling to larger head dims:")
    for dim in [128, 256, 512, 1024, 4096]:
        tq_p = dim * dim + (1 << (bits - 1))
        rq_p = ((dim + 2) // 3) * 4 + sum(len(optimal_centroids(bits - 1, max(((dim+2)//3)*8, 64))) for _ in range(3)) + len(optimal_centroids(max(bits-2, 1), max(((dim+2)//3)*8, 64)))
        print(f"    d={dim:>5d}: TQ={tq_p:>12,d}  RQ={rq_p:>6,d}  ratio={tq_p/rq_p:.0f}x")
    print()


def test_mps_speed():
    """PyTorch MPS benchmark if available."""
    if not HAS_TORCH or not HAS_MPS:
        print("=" * 70)
        print("TEST 6: MPS Speed (SKIPPED — no MPS)")
        print("=" * 70)
        print()
        return

    print("=" * 70)
    print("TEST 6: PyTorch MPS Speed (Apple Silicon)")
    print("=" * 70)

    import torch

    d = 128
    bits = 3
    device = "mps"
    n_warmup = 10
    n_iter = 100

    from turboquant.codebook import optimal_centroids as oc

    # Precompute
    d_eff = max(((d + 2) // 3) * 8, 64)
    centroids = torch.tensor(oc(bits - 1, d_eff), dtype=torch.float32, device=device)

    # Random rotation matrix (TurboQuant style)
    G = torch.randn(d, d, device=device)
    Pi, _ = torch.linalg.qr(G)

    # Rotors (RotorQuant style)
    n_groups = (d + 2) // 3
    rng = np.random.default_rng(42)
    from turboquant.clifford import make_random_rotor
    rotors = []
    for i in range(n_groups):
        r = make_random_rotor(rng)
        rotors.append([r[0], r[4], r[5], r[6]])
    rotors_t = torch.tensor(rotors, dtype=torch.float32, device=device)  # (n_groups, 4)

    print(f"  d={d}, bits={bits}, device={device}\n")

    for n in [1024, 4096, 16384]:
        x = torch.randn(n, d, device=device)
        x = x / x.norm(dim=-1, keepdim=True)

        # TurboQuant: matmul
        torch.mps.synchronize()
        for _ in range(n_warmup):
            y = x @ Pi.T
            idx = (y.unsqueeze(-1) - centroids).abs().argmin(dim=-1)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            y = x @ Pi.T
            idx = (y.unsqueeze(-1) - centroids).abs().argmin(dim=-1)
        torch.mps.synchronize()
        tq_us = (time.perf_counter() - t0) / n_iter * 1e6

        # RotorQuant: embed + rotor sandwich + quantize (PyTorch on MPS)
        torch.mps.synchronize()

        def rq_forward(x_in):
            pad = (3 - d % 3) % 3
            if pad > 0:
                x_in = torch.nn.functional.pad(x_in, (0, pad))
            mv = torch.zeros(x_in.shape[0], n_groups, 8, device=device)
            xg = x_in.reshape(x_in.shape[0], n_groups, 3)
            mv[:, :, 1] = xg[:, :, 0]
            mv[:, :, 2] = xg[:, :, 1]
            mv[:, :, 3] = xg[:, :, 2]

            # Vectorized rotor sandwich
            s = rotors_t[:, 0]    # (n_groups,)
            p12 = rotors_t[:, 1]
            p13 = rotors_t[:, 2]
            p23 = rotors_t[:, 3]

            # Forward GP (sparse)
            t = torch.empty_like(mv)
            t[:,:,0] = s*mv[:,:,0] - p12*mv[:,:,4] - p13*mv[:,:,5] - p23*mv[:,:,6]
            t[:,:,1] = s*mv[:,:,1] + p12*mv[:,:,2] + p13*mv[:,:,3] + p23*mv[:,:,7]
            t[:,:,2] = s*mv[:,:,2] - p12*mv[:,:,1] + p23*mv[:,:,3] - p13*mv[:,:,7]
            t[:,:,3] = s*mv[:,:,3] - p13*mv[:,:,1] - p23*mv[:,:,2] + p12*mv[:,:,7]
            t[:,:,4] = s*mv[:,:,4] + p12*mv[:,:,0]
            t[:,:,5] = s*mv[:,:,5] + p13*mv[:,:,0]
            t[:,:,6] = s*mv[:,:,6] + p23*mv[:,:,0]
            t[:,:,7] = s*mv[:,:,7] - p23*mv[:,:,1] + p13*mv[:,:,2] - p12*mv[:,:,3]

            # Reverse GP (negate bivectors)
            r = torch.empty_like(t)
            r[:,:,0] = s*t[:,:,0] + p12*t[:,:,4] + p13*t[:,:,5] + p23*t[:,:,6]
            r[:,:,1] = s*t[:,:,1] - p12*t[:,:,2] - p13*t[:,:,3] - p23*t[:,:,7]
            r[:,:,2] = s*t[:,:,2] + p12*t[:,:,1] - p23*t[:,:,3] + p13*t[:,:,7]
            r[:,:,3] = s*t[:,:,3] + p13*t[:,:,1] + p23*t[:,:,2] - p12*t[:,:,7]
            r[:,:,4] = s*t[:,:,4] - p12*t[:,:,0]
            r[:,:,5] = s*t[:,:,5] - p13*t[:,:,0]
            r[:,:,6] = s*t[:,:,6] - p23*t[:,:,0]
            r[:,:,7] = s*t[:,:,7] + p23*t[:,:,1] - p13*t[:,:,2] + p12*t[:,:,3]

            # Quantize
            flat = r.reshape(r.shape[0], -1)
            idx = (flat.unsqueeze(-1) - centroids).abs().argmin(dim=-1)
            return idx

        for _ in range(n_warmup):
            rq_forward(x)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            rq_forward(x)
        torch.mps.synchronize()
        rq_us = (time.perf_counter() - t0) / n_iter * 1e6

        def fmt(us):
            if us < 1000: return f"{us:.0f} us"
            return f"{us/1000:.2f} ms"

        ratio = tq_us / rq_us if rq_us > 0 else 0
        faster = "RQ" if ratio > 1 else "TQ"
        print(f"  n={n:>6d}: TQ={fmt(tq_us):>10s}  RQ={fmt(rq_us):>10s}  "
              f"({faster} {max(ratio, 1/ratio):.1f}x faster)")

    print()


if __name__ == "__main__":
    from turboquant.codebook import optimal_centroids

    print()
    print("RotorQuant vs TurboQuant Benchmark")
    print(f"Platform: Apple Silicon Mac Mini M4")
    print()

    test_mse_distortion()
    test_inner_product()
    test_needle()
    test_speed()
    test_params()
    test_mps_speed()

    print("=" * 70)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 70)
