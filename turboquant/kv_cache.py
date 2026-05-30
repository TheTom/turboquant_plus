"""KV Cache integration layer for TurboQuant.

Compresses transformer KV cache tensors using TurboQuant (for K cache, inner product
preservation) and PolarQuant MSE-only (for V cache, MSE preservation).

KV cache shape: (num_layers, num_heads, seq_len, head_dim)
Quantization is along head_dim — each (head_dim,) vector is quantized independently.
"""

import numpy as np
from dataclasses import dataclass, field

from turboquant.turboquant import TurboQuant, TurboQuantMSE, CompressedVector


@dataclass
class CompressedKVCache:
    """Container for a compressed KV cache."""
    # Per-layer, per-head compressed K vectors
    k_compressed: list[list[CompressedVector]] = field(default_factory=list)
    # Per-layer, per-head compressed V (indices + norms)
    v_indices: list[list[np.ndarray]] = field(default_factory=list)
    v_norms: list[list[np.ndarray]] = field(default_factory=list)

    num_layers: int = 0
    num_heads: int = 0
    seq_len: int = 0
    head_dim: int = 0
    k_bit_width: int = 0
    v_bit_width: int = 0


class KVCacheCompressor:
    """Compress and decompress transformer KV cache tensors.

    Uses:
    - TurboQuant (Algorithm 2) for K cache — inner product preservation matters
      for attention score computation (Q @ K^T)
    - TurboQuantMSE (Algorithm 1) for V cache — MSE preservation matters
      for value reconstruction (attn_weights @ V)

    Usage:
        compressor = KVCacheCompressor(head_dim=128, k_bits=3, v_bits=3)

        # Compress
        compressed = compressor.compress(k_cache, v_cache)

        # Decompress
        k_hat, v_hat = compressor.decompress(compressed)

        # Or compress streaming (one token at a time)
        compressor.compress_token(k_vec, v_vec, layer=0, head=0)
    """

    def __init__(
        self,
        head_dim: int,
        k_bits: int = 3,
        v_bits: int = 3,
        seed: int = 42,
        norm_correction: bool = True,
    ):
        """
        Args:
            head_dim: Dimension of each attention head vector.
            k_bits: Bit-width for K cache (TurboQuant, inner product).
            v_bits: Bit-width for V cache (PolarQuant MSE-only).
            seed: Random seed.
        """
        self.head_dim = head_dim
        self.k_bits = k_bits
        self.v_bits = v_bits

        # K cache uses full TurboQuant (inner product preservation)
        self.k_quantizer = TurboQuant(
            head_dim, bit_width=k_bits, seed=seed, norm_correction=norm_correction,
        )

        # V cache uses MSE-only PolarQuant (value reconstruction)
        self.v_quantizer = TurboQuantMSE(
            head_dim, bit_width=v_bits, seed=seed + 500, norm_correction=norm_correction,
        )

    def compress(self, k_cache: np.ndarray, v_cache: np.ndarray) -> CompressedKVCache:
        """Compress full KV cache tensors.

        Args:
            k_cache: Key cache, shape (num_layers, num_heads, seq_len, head_dim).
            v_cache: Value cache, same shape.

        Returns:
            CompressedKVCache with compressed K and V.
        """
        num_layers, num_heads, seq_len, head_dim = k_cache.shape
        assert head_dim == self.head_dim
        assert v_cache.shape == k_cache.shape

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
                # K: batch quantize all seq positions for this layer/head
                k_vecs = k_cache[layer, head]  # (seq_len, head_dim)
                k_compressed = self.k_quantizer.quantize(k_vecs)
                k_layer.append(k_compressed)

                # V: MSE quantize (returns indices + norms)
                v_vecs = v_cache[layer, head]  # (seq_len, head_dim)
                v_indices, v_norms = self.v_quantizer.quantize(v_vecs)
                v_layer_idx.append(v_indices)
                v_layer_norms.append(v_norms)

            result.k_compressed.append(k_layer)
            result.v_indices.append(v_layer_idx)
            result.v_norms.append(v_layer_norms)

        return result

    def decompress(self, compressed: CompressedKVCache) -> tuple[np.ndarray, np.ndarray]:
        """Decompress back to full KV cache tensors.

        Returns:
            (k_cache, v_cache) both shape (num_layers, num_heads, seq_len, head_dim).
        """
        k_cache = np.zeros((
            compressed.num_layers, compressed.num_heads,
            compressed.seq_len, compressed.head_dim
        ))
        v_cache = np.zeros_like(k_cache)

        for layer in range(compressed.num_layers):
            for head in range(compressed.num_heads):
                k_cache[layer, head] = self.k_quantizer.dequantize(
                    compressed.k_compressed[layer][head]
                )
                v_cache[layer, head] = self.v_quantizer.dequantize(
                    compressed.v_indices[layer][head],
                    compressed.v_norms[layer][head],
                )

        return k_cache, v_cache

    def memory_stats(self, seq_len: int, num_layers: int, num_heads: int) -> dict:
        """Compute memory usage statistics.

        Returns dict with original_mb, compressed_mb, ratio.
        """
        n_vectors = num_layers * num_heads * seq_len
        # Original KV cache stores both K and V in fp16.
        original_bytes = n_vectors * self.head_dim * 2 * 2

        # K uses full TurboQuant:
        # - d * k_bits total quantized bits
        # - 32-bit vector norm
        # - 32-bit residual norm
        k_bits_total = n_vectors * (self.head_dim * self.k_bits + 64)
        # V uses MSE-only PolarQuant:
        # - d * v_bits quantized bits
        # - 32-bit vector norm
        v_bits_total = n_vectors * (self.head_dim * self.v_bits + 32)

        compressed_bytes = (k_bits_total + v_bits_total) / 8

        return {
            "original_mb": original_bytes / 1024 / 1024,
            "compressed_mb": compressed_bytes / 1024 / 1024,
            "compression_ratio": original_bytes / compressed_bytes,
            "k_bits_per_value": self.k_bits,
            "v_bits_per_value": self.v_bits,
        }
