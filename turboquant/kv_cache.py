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

    def save(self, path) -> None:
        """Save the compressed cache to a numpy .npz file.

        Args:
            path: File path (string or path-like). A ".npz" extension is
                  appended by numpy if not already present.
        """
        arrays: dict[str, np.ndarray] = {}

        # Metadata scalars stored as 0-d arrays
        arrays["meta_num_layers"] = np.array(self.num_layers)
        arrays["meta_num_heads"] = np.array(self.num_heads)
        arrays["meta_seq_len"] = np.array(self.seq_len)
        arrays["meta_head_dim"] = np.array(self.head_dim)
        arrays["meta_k_bit_width"] = np.array(self.k_bit_width)
        arrays["meta_v_bit_width"] = np.array(self.v_bit_width)

        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                prefix = f"L{layer}_H{head}"
                cv = self.k_compressed[layer][head]
                arrays[f"{prefix}_k_mse_indices"] = np.asarray(cv.mse_indices)
                arrays[f"{prefix}_k_vector_norms"] = np.atleast_1d(
                    np.asarray(cv.vector_norms, dtype=np.float64)
                )
                arrays[f"{prefix}_k_qjl_signs"] = np.asarray(cv.qjl_signs)
                arrays[f"{prefix}_k_residual_norms"] = np.atleast_1d(
                    np.asarray(cv.residual_norms, dtype=np.float64)
                )
                arrays[f"{prefix}_k_bit_width"] = np.array(cv.bit_width)
                arrays[f"{prefix}_v_indices"] = np.asarray(self.v_indices[layer][head])
                arrays[f"{prefix}_v_norms"] = np.atleast_1d(
                    np.asarray(self.v_norms[layer][head], dtype=np.float64)
                )

        np.savez(path, **arrays)

    @classmethod
    def load(cls, path) -> "CompressedKVCache":
        """Load a CompressedKVCache from a numpy .npz file produced by save().

        Args:
            path: File path (string or path-like).

        Returns:
            Reconstructed CompressedKVCache.
        """
        data = np.load(path)

        num_layers = int(data["meta_num_layers"])
        num_heads = int(data["meta_num_heads"])
        seq_len = int(data["meta_seq_len"])
        head_dim = int(data["meta_head_dim"])
        k_bit_width = int(data["meta_k_bit_width"])
        v_bit_width = int(data["meta_v_bit_width"])

        cache = cls(
            num_layers=num_layers,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            k_bit_width=k_bit_width,
            v_bit_width=v_bit_width,
        )

        for layer in range(num_layers):
            k_layer = []
            v_layer_idx = []
            v_layer_norms = []
            for head in range(num_heads):
                prefix = f"L{layer}_H{head}"
                mse_indices = data[f"{prefix}_k_mse_indices"]
                vector_norms_arr = data[f"{prefix}_k_vector_norms"]
                qjl_signs = data[f"{prefix}_k_qjl_signs"]
                residual_norms_arr = data[f"{prefix}_k_residual_norms"]
                bit_width = int(data[f"{prefix}_k_bit_width"])

                # Restore scalar vs array norms depending on shape
                vector_norms = (
                    float(vector_norms_arr[0])
                    if vector_norms_arr.shape == (1,) and mse_indices.ndim == 1
                    else vector_norms_arr
                )
                residual_norms = (
                    float(residual_norms_arr[0])
                    if residual_norms_arr.shape == (1,) and qjl_signs.ndim == 1
                    else residual_norms_arr
                )

                cv = CompressedVector(
                    mse_indices=mse_indices,
                    vector_norms=vector_norms,
                    qjl_signs=qjl_signs,
                    residual_norms=residual_norms,
                    bit_width=bit_width,
                )
                k_layer.append(cv)
                v_layer_idx.append(data[f"{prefix}_v_indices"])
                v_layer_norms.append(data[f"{prefix}_v_norms"])

            cache.k_compressed.append(k_layer)
            cache.v_indices.append(v_layer_idx)
            cache.v_norms.append(v_layer_norms)

        return cache


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

        # Spawn independent child seeds so K and V quantizers use statistically
        # independent random streams without magic offset arithmetic.
        # Accept either an int or an already-created SeedSequence.
        ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        k_child, v_child = ss.spawn(2)

        # K cache uses full TurboQuant (inner product preservation)
        self.k_quantizer = TurboQuant(
            head_dim, bit_width=k_bits, seed=k_child, norm_correction=norm_correction,
        )

        # V cache uses MSE-only PolarQuant (value reconstruction)
        self.v_quantizer = TurboQuantMSE(
            head_dim, bit_width=v_bits, seed=v_child, norm_correction=norm_correction,
        )

        # Streaming buffer: dict[(layer, head)] → list of per-token compressed data.
        # Keys are (layer, head) tuples; values are dicts with 'k' and 'v' lists.
        self._stream_buffer: dict = {}
        self._stream_num_layers: int = 0
        self._stream_num_heads: int = 0

    def compress_token(self, k_vec: np.ndarray, v_vec: np.ndarray, layer: int, head: int) -> None:
        """Compress a single token's K and V vectors and append to the internal buffer.

        Args:
            k_vec: Key vector for this token, shape (head_dim,).
            v_vec: Value vector for this token, shape (head_dim,).
            layer: Layer index.
            head: Head index.
        """
        assert k_vec.shape == (self.head_dim,), (
            f"k_vec shape {k_vec.shape} != ({self.head_dim},)"
        )
        assert v_vec.shape == (self.head_dim,), (
            f"v_vec shape {v_vec.shape} != ({self.head_dim},)"
        )

        key = (layer, head)
        if key not in self._stream_buffer:
            self._stream_buffer[key] = {"k": [], "v_idx": [], "v_norm": []}

        # Quantize K
        k_compressed = self.k_quantizer.quantize(k_vec)
        self._stream_buffer[key]["k"].append(k_compressed)

        # Quantize V
        v_indices, v_norm = self.v_quantizer.quantize(v_vec)
        self._stream_buffer[key]["v_idx"].append(v_indices)
        self._stream_buffer[key]["v_norm"].append(v_norm)

        # Track dimensions
        self._stream_num_layers = max(self._stream_num_layers, layer + 1)
        self._stream_num_heads = max(self._stream_num_heads, head + 1)

    def get_compressed_cache(self) -> "CompressedKVCache":
        """Return the current streaming cache state as a CompressedKVCache.

        Assembles all buffered per-token compressed vectors into the standard
        CompressedKVCache format. The resulting cache can be passed to decompress().

        Returns:
            CompressedKVCache containing all tokens accumulated via compress_token().
        """
        num_layers = self._stream_num_layers
        num_heads = self._stream_num_heads

        if num_layers == 0 or num_heads == 0:
            return CompressedKVCache(
                num_layers=0, num_heads=0, seq_len=0,
                head_dim=self.head_dim,
                k_bit_width=self.k_bits, v_bit_width=self.v_bits,
            )

        # Determine seq_len from the first (layer, head) entry
        first_key = (0, 0)
        seq_len = len(self._stream_buffer.get(first_key, {}).get("k", []))

        result = CompressedKVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=self.head_dim,
            k_bit_width=self.k_bits,
            v_bit_width=self.v_bits,
        )

        for layer in range(num_layers):
            k_layer = []
            v_layer_idx = []
            v_layer_norms = []
            for head in range(num_heads):
                key = (layer, head)
                buf = self._stream_buffer.get(key, {"k": [], "v_idx": [], "v_norm": []})

                # Merge per-token CompressedVectors into a single batched CompressedVector
                token_k_list = buf["k"]
                if token_k_list:
                    merged_k = CompressedVector(
                        mse_indices=np.stack([c.mse_indices for c in token_k_list]),
                        vector_norms=np.stack([c.vector_norms for c in token_k_list]),
                        qjl_signs=np.stack([c.qjl_signs for c in token_k_list]),
                        residual_norms=np.stack([c.residual_norms for c in token_k_list]),
                        bit_width=token_k_list[0].bit_width,
                    )
                else:
                    merged_k = CompressedVector(
                        mse_indices=np.empty((0, self.head_dim), dtype=np.int64),
                        vector_norms=np.empty(0),
                        qjl_signs=np.empty((0, self.head_dim), dtype=np.int8),
                        residual_norms=np.empty(0),
                        bit_width=self.k_bits,
                    )

                k_layer.append(merged_k)
                v_layer_idx.append(
                    np.stack(buf["v_idx"]) if buf["v_idx"] else np.empty((0, self.head_dim))
                )
                v_layer_norms.append(
                    np.array(buf["v_norm"]) if buf["v_norm"] else np.empty(0)
                )

            result.k_compressed.append(k_layer)
            result.v_indices.append(v_layer_idx)
            result.v_norms.append(v_layer_norms)

        return result

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
        original_bytes = n_vectors * self.head_dim * 2  # fp16

        # K: b bits per coord + 32-bit norm
        k_bits_total = n_vectors * (self.head_dim * self.k_bits + 32)
        # V: b bits per coord + 32-bit norm (PolarQuant stores per-vector norm for rescaling)
        v_bits_total = n_vectors * self.head_dim * self.v_bits + n_vectors * 32

        compressed_bytes = (k_bits_total + v_bits_total) / 8

        return {
            "original_mb": original_bytes / 1024 / 1024,
            "compressed_mb": compressed_bytes / 1024 / 1024,
            "compression_ratio": original_bytes / compressed_bytes,
            "k_bits_per_value": self.k_bits,
            "v_bits_per_value": self.v_bits,
        }
