"""Optional MLX backend for the Python TurboQuant prototype.

This module mirrors the existing NumPy prototype with MLX arrays for the
compute-heavy paths. It is intentionally self-contained and import-safe on
systems where MLX is unavailable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from turboquant.codebook import optimal_centroids
from turboquant.kv_cache import CompressedKVCache
from turboquant.qjl import QJL_CONST
from turboquant.rotation import random_rotation_dense
from turboquant.turboquant import CompressedVector

try:  # pragma: no cover - exercised indirectly on MLX-capable systems
    import mlx.core as mx
except ImportError:  # pragma: no cover - expected on non-Apple CI/dev boxes
    mx = None


MLX_AVAILABLE = mx is not None


def _require_mlx() -> None:
    if not MLX_AVAILABLE:
        raise ImportError(
            "MLX backend requires the optional 'mlx' package. "
            "Install it on an MLX-supported system with `pip install mlx`."
        )


def _resolve_dtype(dtype: str):
    _require_mlx()
    try:
        return getattr(mx, dtype)
    except AttributeError as exc:
        raise ValueError(f"Unsupported MLX dtype: {dtype}") from exc


def _is_mlx_array(value: Any) -> bool:
    if not MLX_AVAILABLE:
        return False

    array_type = getattr(mx, "array", None)
    if isinstance(array_type, type):
        return isinstance(value, array_type)

    # Fallback for binding implementations where `mx.array` is callable but not a
    # direct Python type object.
    value_type = type(value)
    return value_type.__module__.startswith("mlx.") and value_type.__name__ == "array"


def _to_mx_array(value: Any, dtype=None):
    _require_mlx()
    if _is_mlx_array(value):
        return value.astype(dtype) if dtype is not None and value.dtype != dtype else value
    return mx.array(value, dtype=dtype)


def to_numpy(value: Any) -> np.ndarray:
    """Convert MLX arrays to NumPy arrays without requiring callers to branch."""
    if _is_mlx_array(value):
        return np.array(value)
    return np.asarray(value)


class MLXPolarQuant:
    """MLX implementation of PolarQuant using the NumPy-calibrated codebook."""

    def __init__(
        self,
        d: int,
        bit_width: int,
        seed: int = 42,
        norm_correction: bool = True,
        dtype: str = "float32",
    ):
        _require_mlx()
        self.d = d
        self.bit_width = bit_width
        self.n_centroids = 1 << bit_width
        self.norm_correction = norm_correction
        self.dtype = _resolve_dtype(dtype)

        rng = np.random.default_rng(seed)
        rotation = random_rotation_dense(d, rng)
        centroids = optimal_centroids(bit_width, d)
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        self.rotation = mx.array(rotation, dtype=self.dtype)
        self.rotation_t = mx.transpose(self.rotation)
        self.centroids = mx.array(centroids, dtype=self.dtype)
        self.boundaries = mx.array(boundaries, dtype=self.dtype)

    def quantize(self, x: Any):
        x_arr = _to_mx_array(x, self.dtype)
        single = x_arr.ndim == 1
        if single:
            x_arr = x_arr.reshape(1, self.d)
        if x_arr.ndim != 2 or x_arr.shape[1] != self.d:
            raise ValueError(f"Expected shape ({self.d},) or (batch, {self.d}), got {x_arr.shape}")

        norms = mx.linalg.norm(x_arr, axis=1)
        safe_norms = mx.where(norms > 0, norms, 1.0)
        x_normalized = x_arr / safe_norms.reshape((-1, 1))

        y = mx.matmul(x_normalized, self.rotation_t)
        gt = (y[..., None] > self.boundaries.reshape((1, 1, self.boundaries.shape[0]))).astype(mx.int32)
        indices = gt.sum(axis=-1).astype(mx.uint32)

        if single:
            return mx.squeeze(indices, axis=0), mx.squeeze(norms, axis=0)
        return indices, norms

    def dequantize(self, indices: Any, norms: Any):
        indices_arr = _to_mx_array(indices)
        norms_arr = _to_mx_array(norms, self.dtype)

        single = indices_arr.ndim == 1
        if single:
            indices_arr = indices_arr.reshape(1, self.d)
            norms_arr = norms_arr.reshape((1,))
        if indices_arr.ndim != 2 or indices_arr.shape[1] != self.d:
            raise ValueError(
                f"Expected index shape ({self.d},) or (batch, {self.d}), got {indices_arr.shape}"
            )

        y_hat = mx.take(self.centroids, indices_arr)

        if self.norm_correction:
            y_hat_norms = mx.linalg.norm(y_hat, axis=1, keepdims=True)
            y_hat_norms = mx.where(y_hat_norms > 1e-10, y_hat_norms, 1.0)
            y_hat = y_hat / y_hat_norms

        x_hat_unit = mx.matmul(y_hat, self.rotation)
        x_hat = x_hat_unit * norms_arr.reshape((-1, 1))

        return mx.squeeze(x_hat, axis=0) if single else x_hat

    def quantize_and_residual(self, x: Any):
        indices, norms = self.quantize(x)
        x_hat = self.dequantize(indices, norms)
        residual = _to_mx_array(x, self.dtype) - x_hat
        return indices, norms, residual


class MLXQJL:
    """MLX implementation of the 1-bit QJL residual stage."""

    def __init__(self, d: int, seed: int = 123, dtype: str = "float32"):
        _require_mlx()
        self.d = d
        self.dtype = _resolve_dtype(dtype)
        rng = np.random.default_rng(seed)
        self.S = mx.array(rng.standard_normal((d, d)), dtype=self.dtype)
        self.S_t = mx.transpose(self.S)

    def quantize(self, r: Any):
        r_arr = _to_mx_array(r, self.dtype)
        single = r_arr.ndim == 1
        if single:
            r_arr = r_arr.reshape(1, self.d)
        if r_arr.ndim != 2 or r_arr.shape[1] != self.d:
            raise ValueError(f"Expected shape ({self.d},) or (batch, {self.d}), got {r_arr.shape}")

        norms = mx.linalg.norm(r_arr, axis=1)
        projected = mx.matmul(r_arr, self.S_t)
        signs = mx.where(projected >= 0, 1, -1).astype(mx.int8)

        if single:
            return mx.squeeze(signs, axis=0), mx.squeeze(norms, axis=0)
        return signs, norms

    def dequantize(self, signs: Any, norms: Any):
        signs_arr = _to_mx_array(signs)
        norms_arr = _to_mx_array(norms, self.dtype)

        single = signs_arr.ndim == 1
        if single:
            signs_arr = signs_arr.reshape(1, self.d)
            norms_arr = norms_arr.reshape((1,))
        if signs_arr.ndim != 2 or signs_arr.shape[1] != self.d:
            raise ValueError(
                f"Expected sign shape ({self.d},) or (batch, {self.d}), got {signs_arr.shape}"
            )

        reconstructed = mx.matmul(signs_arr.astype(self.dtype), self.S)
        scale = norms_arr * (QJL_CONST / self.d)
        reconstructed = reconstructed * scale.reshape((-1, 1))

        return mx.squeeze(reconstructed, axis=0) if single else reconstructed


class MLXTurboQuant:
    """Full MLX TurboQuant path for K-cache parity with the NumPy prototype."""

    def __init__(
        self,
        d: int,
        bit_width: int,
        seed: int = 42,
        norm_correction: bool = True,
        dtype: str = "float32",
    ):
        if bit_width < 2:
            raise ValueError(
                "TurboQuant requires bit_width >= 2 (1 bit PolarQuant + 1 bit QJL). "
                "For 1-bit, use QJL directly."
            )

        self.d = d
        self.bit_width = bit_width
        self.polar_quant = MLXPolarQuant(
            d,
            bit_width=bit_width - 1,
            seed=seed,
            norm_correction=norm_correction,
            dtype=dtype,
        )
        self.qjl = MLXQJL(d, seed=seed + 1000, dtype=dtype)

    def quantize(self, x: Any) -> CompressedVector:
        mse_indices, vector_norms, residual = self.polar_quant.quantize_and_residual(x)
        qjl_signs, residual_norms = self.qjl.quantize(residual)
        return CompressedVector(
            mse_indices=mse_indices,
            vector_norms=vector_norms,
            qjl_signs=qjl_signs,
            residual_norms=residual_norms,
            bit_width=self.bit_width,
        )

    def dequantize(self, compressed: CompressedVector):
        x_mse = self.polar_quant.dequantize(compressed.mse_indices, compressed.vector_norms)
        x_qjl = self.qjl.dequantize(compressed.qjl_signs, compressed.residual_norms)
        return x_mse + x_qjl

    def compressed_size_bits(self, n_vectors: int) -> int:
        per_vector = self.d * self.bit_width
        norms = 32
        return n_vectors * (per_vector + norms)

    def compression_ratio(self, original_bits_per_value: int = 16) -> float:
        original_per_vector = self.d * original_bits_per_value
        compressed_per_vector = self.d * self.bit_width + 32
        return original_per_vector / compressed_per_vector


class MLXTurboQuantMSE:
    """MSE-only MLX TurboQuant path for V-cache compression."""

    def __init__(
        self,
        d: int,
        bit_width: int,
        seed: int = 42,
        norm_correction: bool = True,
        dtype: str = "float32",
    ):
        self.d = d
        self.bit_width = bit_width
        self.polar_quant = MLXPolarQuant(
            d,
            bit_width=bit_width,
            seed=seed,
            norm_correction=norm_correction,
            dtype=dtype,
        )

    def quantize(self, x: Any):
        return self.polar_quant.quantize(x)

    def dequantize(self, indices: Any, norms: Any):
        return self.polar_quant.dequantize(indices, norms)


class MLXKVCacheCompressor:
    """MLX equivalent of the NumPy KV cache compressor."""

    def __init__(
        self,
        head_dim: int,
        k_bits: int = 3,
        v_bits: int = 3,
        seed: int = 42,
        norm_correction: bool = True,
        dtype: str = "float32",
    ):
        self.head_dim = head_dim
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.dtype = dtype

        self.k_quantizer = MLXTurboQuant(
            head_dim,
            bit_width=k_bits,
            seed=seed,
            norm_correction=norm_correction,
            dtype=dtype,
        )
        self.v_quantizer = MLXTurboQuantMSE(
            head_dim,
            bit_width=v_bits,
            seed=seed + 500,
            norm_correction=norm_correction,
            dtype=dtype,
        )

    def compress(self, k_cache: Any, v_cache: Any) -> CompressedKVCache:
        k_cache_arr = _to_mx_array(k_cache, _resolve_dtype(self.dtype))
        v_cache_arr = _to_mx_array(v_cache, _resolve_dtype(self.dtype))

        if k_cache_arr.shape != v_cache_arr.shape:
            raise ValueError(f"K/V cache shapes must match, got {k_cache_arr.shape} and {v_cache_arr.shape}")
        if k_cache_arr.ndim != 4:
            raise ValueError(f"Expected KV cache shape (layers, heads, seq, dim), got {k_cache_arr.shape}")

        num_layers, num_heads, seq_len, head_dim = k_cache_arr.shape
        if head_dim != self.head_dim:
            raise ValueError(f"Expected head_dim={self.head_dim}, got {head_dim}")

        result = CompressedKVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            k_bit_width=self.k_bits,
            v_bit_width=self.v_bits,
        )

        for layer in range(num_layers):
            k_layer = []
            v_layer_idx = []
            v_layer_norms = []
            for head in range(num_heads):
                k_compressed = self.k_quantizer.quantize(k_cache_arr[layer, head])
                v_indices, v_norms = self.v_quantizer.quantize(v_cache_arr[layer, head])
                k_layer.append(k_compressed)
                v_layer_idx.append(v_indices)
                v_layer_norms.append(v_norms)

            result.k_compressed.append(k_layer)
            result.v_indices.append(v_layer_idx)
            result.v_norms.append(v_layer_norms)

        return result

    def decompress(self, compressed: CompressedKVCache):
        k_layers = []
        v_layers = []

        for layer in range(compressed.num_layers):
            k_heads = []
            v_heads = []
            for head in range(compressed.num_heads):
                k_heads.append(self.k_quantizer.dequantize(compressed.k_compressed[layer][head]))
                v_heads.append(
                    self.v_quantizer.dequantize(
                        compressed.v_indices[layer][head],
                        compressed.v_norms[layer][head],
                    )
                )

            k_layers.append(mx.stack(k_heads, axis=0))
            v_layers.append(mx.stack(v_heads, axis=0))

        return mx.stack(k_layers, axis=0), mx.stack(v_layers, axis=0)

    def memory_stats(self, seq_len: int, num_layers: int, num_heads: int) -> dict[str, float]:
        n_vectors = num_layers * num_heads * seq_len
        original_bytes = n_vectors * self.head_dim * 2
        k_bits_total = n_vectors * (self.head_dim * self.k_bits + 32)
        v_bits_total = n_vectors * self.head_dim * self.v_bits
        compressed_bytes = (k_bits_total + v_bits_total) / 8

        return {
            "original_mb": original_bytes / 1024 / 1024,
            "compressed_mb": compressed_bytes / 1024 / 1024,
            "compression_ratio": original_bytes / compressed_bytes,
            "k_bits_per_value": self.k_bits,
            "v_bits_per_value": self.v_bits,
        }
