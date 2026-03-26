"""Layer-adaptive KV cache compression.

Key finding from TurboQuant paper: the last 8/40 layers account for nearly ALL
quality loss when using aggressive quantization. This module provides per-layer
bit-width configuration so that sensitive layers (typically the last ~20%) use
higher precision (e.g., 8-bit) while early layers use aggressive TurboQuant
(e.g., 3-bit).

Mode 2 from the paper:
  - Layers 0-31: turbo3 (3-bit TurboQuant)
  - Layers 32-39: q8_0  (8-bit quantization)
"""

import numpy as np
from dataclasses import dataclass, field

from turboquant.turboquant import TurboQuant, TurboQuantMSE, CompressedVector
from turboquant.kv_cache import KVCacheCompressor, CompressedKVCache


# ---------------------------------------------------------------------------
# Default presets
# ---------------------------------------------------------------------------

def default_40layer_config() -> dict[int, int]:
    """Default config for a 40-layer model (paper Mode 2).

    Layers 0-31: 3-bit, layers 32-39: 8-bit.
    """
    config: dict[int, int] = {}
    for i in range(32):
        config[i] = 3
    for i in range(32, 40):
        config[i] = 8
    return config


def make_layer_config(
    total_layers: int,
    default_bits: int = 3,
    high_bits: int = 8,
    high_frac: float = 0.2,
) -> dict[int, int]:
    """Build a layer config where the last ``high_frac`` layers get ``high_bits``.

    Args:
        total_layers: Number of transformer layers.
        default_bits: Bit-width for early layers.
        high_bits: Bit-width for late (sensitive) layers.
        high_frac: Fraction of layers at the end that use high_bits.

    Returns:
        Mapping from layer index to bit-width.
    """
    cutoff = int(total_layers * (1.0 - high_frac))
    config: dict[int, int] = {}
    for i in range(total_layers):
        config[i] = default_bits if i < cutoff else high_bits
    return config


# ---------------------------------------------------------------------------
# Compressed container
# ---------------------------------------------------------------------------

@dataclass
class CompressedLayerAdaptiveKVCache:
    """Container for a layer-adaptive compressed KV cache."""
    # Per-layer CompressedKVCache (each may have different bit-width)
    layer_caches: list[CompressedKVCache] = field(default_factory=list)

    num_layers: int = 0
    num_heads: int = 0
    seq_len: int = 0
    head_dim: int = 0
    layers_config: dict[int, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main compressor
# ---------------------------------------------------------------------------

class LayerAdaptiveCompressor:
    """KV cache compressor with per-layer bit-width configuration.

    Wraps ``KVCacheCompressor`` with one compressor per unique bit-width,
    dispatching each layer to the appropriate compressor.

    Usage::

        config = make_layer_config(total_layers=40, default_bits=3,
                                   high_bits=8, high_frac=0.2)
        compressor = LayerAdaptiveCompressor(head_dim=128, layers_config=config)
        compressed = compressor.compress(k_cache, v_cache)
        k_hat, v_hat = compressor.decompress(compressed)
        print(compressor.effective_compression_ratio())
    """

    def __init__(
        self,
        head_dim: int,
        layers_config: dict[int, int],
        v_bits_override: dict[int, int] | None = None,
        seed: int = 42,
    ):
        """
        Args:
            head_dim: Dimension of each attention head.
            layers_config: Mapping layer_index -> bit_width (used for both K and V
                unless ``v_bits_override`` is given).
            v_bits_override: Optional per-layer V bit-width override.  If not
                provided, V uses the same bit-width as K for each layer.
            seed: Random seed.
        """
        self.head_dim = head_dim
        self.layers_config = dict(layers_config)
        self.v_bits_override = dict(v_bits_override) if v_bits_override else {}
        self.seed = seed

        # Build one compressor per unique (k_bits, v_bits) pair, keyed by tuple
        self._compressors: dict[tuple[int, int], KVCacheCompressor] = {}
        for layer_idx, k_bits in self.layers_config.items():
            v_bits = self.v_bits_override.get(layer_idx, k_bits)
            key = (k_bits, v_bits)
            if key not in self._compressors:
                self._compressors[key] = KVCacheCompressor(
                    head_dim=head_dim,
                    k_bits=k_bits,
                    v_bits=v_bits,
                    seed=seed,
                )

    def _get_compressor(self, layer_idx: int) -> KVCacheCompressor:
        k_bits = self.layers_config[layer_idx]
        v_bits = self.v_bits_override.get(layer_idx, k_bits)
        return self._compressors[(k_bits, v_bits)]

    # ------------------------------------------------------------------
    # Compress / decompress
    # ------------------------------------------------------------------

    def compress(
        self, k_cache: np.ndarray, v_cache: np.ndarray,
    ) -> CompressedLayerAdaptiveKVCache:
        """Compress full KV cache with per-layer bit-widths.

        Args:
            k_cache: Key cache, shape (num_layers, num_heads, seq_len, head_dim).
            v_cache: Value cache, same shape.

        Returns:
            ``CompressedLayerAdaptiveKVCache`` containing per-layer compressed data.
        """
        num_layers, num_heads, seq_len, head_dim = k_cache.shape
        assert head_dim == self.head_dim
        assert v_cache.shape == k_cache.shape

        # Validate that config covers all layers
        for layer_idx in range(num_layers):
            if layer_idx not in self.layers_config:
                raise ValueError(
                    f"Layer {layer_idx} not in layers_config. "
                    f"Config covers layers: {sorted(self.layers_config.keys())}"
                )

        result = CompressedLayerAdaptiveKVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            layers_config=dict(self.layers_config),
        )

        for layer_idx in range(num_layers):
            compressor = self._get_compressor(layer_idx)
            # Wrap single layer in the 4D shape expected by KVCacheCompressor
            k_layer = k_cache[layer_idx:layer_idx + 1]  # (1, heads, seq, dim)
            v_layer = v_cache[layer_idx:layer_idx + 1]
            compressed_layer = compressor.compress(k_layer, v_layer)
            result.layer_caches.append(compressed_layer)

        return result

    def decompress(
        self, compressed: CompressedLayerAdaptiveKVCache,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decompress back to full KV cache tensors.

        Returns:
            (k_cache, v_cache) both shape (num_layers, num_heads, seq_len, head_dim).
        """
        k_layers = []
        v_layers = []

        for layer_idx, layer_cache in enumerate(compressed.layer_caches):
            compressor = self._get_compressor(layer_idx)
            k_layer, v_layer = compressor.decompress(layer_cache)
            k_layers.append(k_layer)
            v_layers.append(v_layer)

        return np.concatenate(k_layers, axis=0), np.concatenate(v_layers, axis=0)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def effective_bits_per_value(self) -> float:
        """Compute the weighted-average bits per value across all layers.

        Returns:
            Average bit-width (K and V averaged).
        """
        total_layers = len(self.layers_config)
        if total_layers == 0:
            return 0.0

        total_k_bits = 0.0
        total_v_bits = 0.0
        for layer_idx, k_bits in self.layers_config.items():
            v_bits = self.v_bits_override.get(layer_idx, k_bits)
            total_k_bits += k_bits
            total_v_bits += v_bits

        avg_k = total_k_bits / total_layers
        avg_v = total_v_bits / total_layers
        return (avg_k + avg_v) / 2.0

    def effective_compression_ratio(self, original_bits: int = 16) -> float:
        """Compute effective compression ratio vs original precision.

        Args:
            original_bits: Bits per value in the original cache (16 for fp16).

        Returns:
            Compression ratio (e.g., 4.0 means 4x smaller).
        """
        avg_bits = self.effective_bits_per_value()
        if avg_bits == 0:
            return float("inf")
        return original_bits / avg_bits

    def layer_summary(self) -> list[dict]:
        """Return a per-layer summary of bit-width configuration.

        Returns:
            List of dicts with layer_idx, k_bits, v_bits.
        """
        summary = []
        for layer_idx in sorted(self.layers_config.keys()):
            k_bits = self.layers_config[layer_idx]
            v_bits = self.v_bits_override.get(layer_idx, k_bits)
            summary.append({
                "layer": layer_idx,
                "k_bits": k_bits,
                "v_bits": v_bits,
            })
        return summary
