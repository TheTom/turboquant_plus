"""Temporal decay configuration for KV cache compression.

Tokens that are far in the past contribute less to attention and can be
compressed more aggressively. This module provides configuration and logic
for mapping token age to bit-width, with optional layer-awareness (early
layers decay faster than late layers).

The actual llama.cpp C integration is blocked; this is the Python design/config
layer with complete logic for bit-width selection and simulated compression.
"""

import numpy as np
from dataclasses import dataclass

from turboquant.turboquant import TurboQuant, TurboQuantMSE


@dataclass
class TemporalDecayConfig:
    """Configuration for temporal-decay-aware quantization.

    Attributes:
        recent_bits: Bit-width for recently generated tokens.
        old_bits: Bit-width for old (past threshold) tokens.
        decay_threshold: Token age (in steps) at which we switch from
            recent_bits to old_bits.
        layer_aware: If True, early layers (first 80%) decay faster
            to old_bits, while late layers (last 20%) stay at recent_bits.
    """
    recent_bits: int = 3
    old_bits: int = 2
    decay_threshold: int = 256
    layer_aware: bool = True


class TemporalDecayCompressor:
    """Maps token age to bit-width and compresses accordingly.

    Usage::

        config = TemporalDecayConfig(recent_bits=3, old_bits=2,
                                     decay_threshold=256, layer_aware=True)
        tdc = TemporalDecayCompressor(head_dim=128, config=config)

        bits = tdc.get_bits_for_token(age=300, layer=0, total_layers=40)
        result = tdc.compress_with_decay(keys, values, token_ages,
                                         layer_idx=5, total_layers=40)
    """

    def __init__(self, head_dim: int, config: TemporalDecayConfig | None = None,
                 seed: int = 42):
        """
        Args:
            head_dim: Dimension of each attention head vector.
            config: Temporal decay configuration. Uses defaults if None.
            seed: Random seed for quantizers.
        """
        self.head_dim = head_dim
        self.config = config or TemporalDecayConfig()
        self.seed = seed

        # Build quantizers for each unique bit-width we might need
        self._k_quantizers: dict[int, TurboQuant] = {}
        self._v_quantizers: dict[int, TurboQuantMSE] = {}
        for bits in {self.config.recent_bits, self.config.old_bits}:
            if bits >= 2:
                self._k_quantizers[bits] = TurboQuant(head_dim, bit_width=bits, seed=seed)
            self._v_quantizers[bits] = TurboQuantMSE(head_dim, bit_width=bits, seed=seed + 500)

    def get_bits_for_token(self, age: int, layer: int, total_layers: int) -> int:
        """Determine bit-width for a token given its age and layer position.

        Args:
            age: Token age in steps (0 = most recent).
            layer: Layer index (0-based).
            total_layers: Total number of layers in the model.

        Returns:
            Bit-width to use for this token at this layer.
        """
        cfg = self.config

        if not cfg.layer_aware:
            # Simple threshold: recent vs old
            return cfg.recent_bits if age < cfg.decay_threshold else cfg.old_bits

        # Layer-aware mode:
        # Late layers (last 20%) always keep recent_bits
        late_cutoff = int(total_layers * 0.8)
        if layer >= late_cutoff:
            return cfg.recent_bits

        # Early layers (first 80%) decay faster
        # Use a reduced threshold: scale linearly with position in early range
        # Layer 0 decays at 50% of threshold, layer (late_cutoff-1) at 100%
        if late_cutoff <= 0:
            scale = 1.0
        else:
            scale = 0.5 + 0.5 * (layer / late_cutoff)
        effective_threshold = int(cfg.decay_threshold * scale)

        return cfg.recent_bits if age < effective_threshold else cfg.old_bits

    def get_bits_map(
        self, token_ages: np.ndarray, layer_idx: int, total_layers: int,
    ) -> np.ndarray:
        """Compute bit-width for each token in a sequence.

        Args:
            token_ages: 1D array of token ages, shape (seq_len,).
            layer_idx: Current layer index.
            total_layers: Total layers in the model.

        Returns:
            1D int array of bit-widths, shape (seq_len,).
        """
        return np.array([
            self.get_bits_for_token(int(age), layer_idx, total_layers)
            for age in token_ages
        ], dtype=np.int32)

    def compress_with_decay(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        token_ages: np.ndarray,
        layer_idx: int,
        total_layers: int,
    ) -> dict:
        """Compress keys and values with age-dependent bit-widths.

        Groups tokens by their assigned bit-width, compresses each group
        with the appropriate quantizer, then returns a dict with the
        compressed data and metadata.

        Args:
            keys: Key vectors, shape (seq_len, head_dim).
            values: Value vectors, shape (seq_len, head_dim).
            token_ages: 1D array of ages, shape (seq_len,).
            layer_idx: Current layer index.
            total_layers: Total number of layers.

        Returns:
            Dict with keys:
                - ``bits_map``: per-token bit-widths
                - ``groups``: list of dicts, each with ``bits``, ``indices``,
                  ``k_compressed``, ``v_indices``, ``v_norms``
                - ``seq_len``, ``head_dim``, ``layer_idx``
        """
        seq_len, head_dim = keys.shape
        assert head_dim == self.head_dim
        assert values.shape == keys.shape
        assert len(token_ages) == seq_len

        bits_map = self.get_bits_map(token_ages, layer_idx, total_layers)
        unique_bits = np.unique(bits_map)

        groups = []
        for bits in unique_bits:
            bits = int(bits)
            mask = bits_map == bits
            token_indices = np.where(mask)[0]

            if len(token_indices) == 0:
                continue

            k_group = keys[token_indices]  # (n, head_dim)
            v_group = values[token_indices]

            # Compress K
            k_quantizer = self._k_quantizers.get(bits)
            if k_quantizer is not None:
                k_compressed = k_quantizer.quantize(k_group)
            else:
                # For 1-bit (no TurboQuant), fall back to MSE-only
                k_compressed = None

            # Compress V
            v_quantizer = self._v_quantizers[bits]
            v_indices, v_norms = v_quantizer.quantize(v_group)

            groups.append({
                "bits": bits,
                "token_indices": token_indices,
                "k_compressed": k_compressed,
                "v_indices": v_indices,
                "v_norms": v_norms,
            })

        return {
            "bits_map": bits_map,
            "groups": groups,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "layer_idx": layer_idx,
        }

    def decompress_with_decay(self, compressed: dict) -> tuple[np.ndarray, np.ndarray]:
        """Decompress data produced by ``compress_with_decay``.

        Returns:
            (keys, values) both shape (seq_len, head_dim).
        """
        seq_len = compressed["seq_len"]
        head_dim = compressed["head_dim"]

        keys_out = np.zeros((seq_len, head_dim))
        values_out = np.zeros((seq_len, head_dim))

        for group in compressed["groups"]:
            bits = group["bits"]
            indices = group["token_indices"]

            # Decompress K
            k_compressed = group["k_compressed"]
            if k_compressed is not None:
                k_quantizer = self._k_quantizers[bits]
                k_recon = k_quantizer.dequantize(k_compressed)
            else:
                k_recon = np.zeros((len(indices), head_dim))

            # Decompress V
            v_quantizer = self._v_quantizers[bits]
            v_recon = v_quantizer.dequantize(group["v_indices"], group["v_norms"])

            keys_out[indices] = k_recon
            values_out[indices] = v_recon

        return keys_out, values_out

    def memory_savings_estimate(
        self,
        token_ages: np.ndarray,
        layer_idx: int,
        total_layers: int,
        original_bits: int = 16,
    ) -> dict:
        """Estimate memory savings for a given token age distribution.

        Returns:
            Dict with original_bits_total, compressed_bits_total, ratio.
        """
        bits_map = self.get_bits_map(token_ages, layer_idx, total_layers)
        seq_len = len(token_ages)

        original_total = seq_len * self.head_dim * original_bits * 2  # K + V
        compressed_total = 0
        for bits in np.unique(bits_map):
            n_tokens = int(np.sum(bits_map == bits))
            # K: bits per coord + 32-bit norm per vector
            # V: bits per coord
            k_bits = n_tokens * (self.head_dim * int(bits) + 32)
            v_bits = n_tokens * self.head_dim * int(bits)
            compressed_total += k_bits + v_bits

        return {
            "original_bits": original_total,
            "compressed_bits": compressed_total,
            "ratio": original_total / compressed_total if compressed_total > 0 else float("inf"),
            "avg_bits": float(np.mean(bits_map)),
        }
