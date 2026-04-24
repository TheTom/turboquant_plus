"""Benchmark the optional MLX backend against the NumPy prototype.

Usage:
    python3 benchmarks/benchmark_mlx_backend.py
"""

from __future__ import annotations

import time

import numpy as np

from turboquant import KVCacheCompressor, MLXKVCacheCompressor, MLX_AVAILABLE
from turboquant.mlx_backend import to_numpy


QWEN_27B = {
    "name": "Qwen 3.5 27B (dense)",
    "num_layers": 28,
    "num_kv_heads": 8,
    "head_dim": 128,
}

QWEN_MOE = {
    "name": "Qwen 3.5 35B-A3B (MoE)",
    "num_layers": 28,
    "num_kv_heads": 8,
    "head_dim": 128,
}


def simulate_kv_cache(config: dict, seq_len: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape = (config["num_layers"], config["num_kv_heads"], seq_len, config["head_dim"])
    scale = 1.0 / np.sqrt(config["head_dim"])
    k_cache = rng.standard_normal(shape) * scale
    v_cache = rng.standard_normal(shape) * scale
    return k_cache, v_cache


def run_backend(name: str, compressor, k_cache: np.ndarray, v_cache: np.ndarray):
    t0 = time.perf_counter()
    compressed = compressor.compress(k_cache, v_cache)
    k_hat, v_hat = compressor.decompress(compressed)

    if MLX_AVAILABLE and name == "mlx":
        import mlx.core as mx

        mx.eval(k_hat, v_hat)

    elapsed = time.perf_counter() - t0
    k_hat_np = to_numpy(k_hat)
    v_hat_np = to_numpy(v_hat)

    k_mse = np.mean((k_cache - k_hat_np) ** 2)
    v_mse = np.mean((v_cache - v_hat_np) ** 2)
    return elapsed, k_mse, v_mse


def main():
    if not MLX_AVAILABLE:
        raise SystemExit("MLX is not available. Install `mlx` on an MLX-supported system to run this benchmark.")

    print("=" * 70)
    print("TURBOQUANT MLX BACKEND BENCHMARK")
    print("=" * 70)

    for config in (QWEN_27B, QWEN_MOE):
        print(f"\n{config['name']}")
        for seq_len in (512, 2048):
            print(f"  seq_len={seq_len}")
            k_cache, v_cache = simulate_kv_cache(config, seq_len)

            numpy_compressor = KVCacheCompressor(head_dim=config["head_dim"], k_bits=3, v_bits=3)
            mlx_compressor = MLXKVCacheCompressor(
                head_dim=config["head_dim"],
                k_bits=3,
                v_bits=3,
                dtype="float32",
            )

            numpy_elapsed, numpy_k_mse, numpy_v_mse = run_backend("numpy", numpy_compressor, k_cache, v_cache)
            mlx_elapsed, mlx_k_mse, mlx_v_mse = run_backend("mlx", mlx_compressor, k_cache, v_cache)

            print(
                f"    NumPy: {numpy_elapsed:.2f}s, "
                f"K MSE={numpy_k_mse:.8f}, V MSE={numpy_v_mse:.8f}"
            )
            print(
                f"    MLX:   {mlx_elapsed:.2f}s, "
                f"K MSE={mlx_k_mse:.8f}, V MSE={mlx_v_mse:.8f}"
            )


if __name__ == "__main__":
    main()
