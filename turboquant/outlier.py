"""Outlier channel strategy for non-integer bit precision.

Paper Section on Non-Integer Bit Precision:
Split channels into outlier (higher bits) and non-outlier (lower bits).

Examples:
- 2.5-bit: 25% channels at 3 bits + 75% at 2 bits → (0.25×3 + 0.75×2) = 2.25 avg
  Adjusted: 32/128 outlier at 3b + 96/128 normal at 2b = (32×3 + 96×2)/128 = 2.5
- 3.5-bit: 64/128 outlier at 4b + 64/128 normal at 3b = (64×4 + 64×3)/128 = 3.5
"""

import numpy as np
from dataclasses import dataclass

from turboquant.polar_quant import PolarQuant
from turboquant.qjl import QJL


@dataclass
class OutlierCompressedVector:
    """Container for outlier-strategy compressed vector."""
    outlier_indices: np.ndarray    # indices for outlier channels (higher bits)
    outlier_norms: np.ndarray      # norms for outlier channels
    normal_indices: np.ndarray     # indices for normal channels (lower bits)
    normal_norms: np.ndarray       # norms for normal channels
    qjl_signs: np.ndarray          # QJL signs for full residual
    residual_norms: np.ndarray     # ||residual||_2
    effective_bits: float


def _compute_channel_split(d: int, target_bits: float) -> tuple[int, int, int, int]:
    """Compute how many channels get higher vs lower bit-width.

    Args:
        d: Total number of channels.
        target_bits: Target average bits per channel (e.g., 2.5, 3.5).

    Returns:
        (n_outlier, outlier_bits, n_normal, normal_bits)
    """
    # TurboQuant uses (b-1) PolarQuant + 1 QJL, so effective PolarQuant bits:
    # For target 2.5: outlier gets 2-bit PQ + 1-bit QJL = 3, normal gets 1-bit PQ + 1-bit QJL = 2
    # For target 3.5: outlier gets 3-bit PQ + 1-bit QJL = 4, normal gets 2-bit PQ + 1-bit QJL = 3

    low_bits = int(np.floor(target_bits))
    high_bits = low_bits + 1
    frac = target_bits - low_bits  # fraction of channels at high_bits

    n_outlier = int(round(d * frac))
    n_normal = d - n_outlier

    return n_outlier, high_bits, n_normal, low_bits


class OutlierTurboQuant:
    """TurboQuant with outlier channel strategy for non-integer bit rates.

    Splits channels into outlier (higher bit-width) and normal (lower bit-width)
    to achieve fractional average bit rates like 2.5 or 3.5 bits per channel.

    Usage:
        oq = OutlierTurboQuant(d=128, target_bits=2.5, seed=42)
        compressed = oq.quantize(x)
        x_hat = oq.dequantize(compressed)
    """

    def __init__(self, d: int, target_bits: float, seed: int = 42):
        self.d = d
        self.target_bits = target_bits

        n_outlier, high_bits, n_normal, low_bits = _compute_channel_split(d, target_bits)
        self.n_outlier = n_outlier
        self.n_normal = n_normal
        self.high_bits = high_bits
        self.low_bits = low_bits

        # Effective bit rate
        self.effective_bits = (n_outlier * high_bits + n_normal * low_bits) / d

        # Channel indices (fixed — outlier channels are the first n_outlier)
        # In practice, you'd pick channels with highest activation magnitude.
        # For data-oblivious quantization (paper's approach), we use fixed split.
        self.outlier_idx = np.arange(n_outlier)
        self.normal_idx = np.arange(n_outlier, d)

        rng = np.random.default_rng(seed)

        # Separate PolarQuant for outlier and normal channels
        # PolarQuant bit-width is (total - 1) since QJL adds 1 bit
        self.pq_outlier = PolarQuant(n_outlier, bit_width=high_bits - 1, seed=seed) if n_outlier > 0 else None
        self.pq_normal = PolarQuant(n_normal, bit_width=low_bits - 1, seed=seed + 500) if n_normal > 0 else None

        # QJL on full residual
        self.qjl = QJL(d, seed=seed + 1000)

    def quantize(self, x: np.ndarray) -> OutlierCompressedVector:
        """Quantize with outlier channel split."""
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]

        batch = x.shape[0]

        # Split channels
        x_outlier = x[:, self.outlier_idx]  # (batch, n_outlier)
        x_normal = x[:, self.normal_idx]    # (batch, n_normal)

        # Quantize outlier channels at higher bits
        if self.pq_outlier is not None:
            out_idx, out_norms, out_residual = self.pq_outlier.quantize_and_residual(x_outlier if batch > 1 else x_outlier[0])
        else:
            out_idx = np.array([])
            out_norms = np.array([])
            out_residual = np.zeros_like(x_outlier)

        # Quantize normal channels at lower bits
        if self.pq_normal is not None:
            norm_idx, norm_norms, norm_residual = self.pq_normal.quantize_and_residual(x_normal if batch > 1 else x_normal[0])
        else:
            norm_idx = np.array([])
            norm_norms = np.array([])
            norm_residual = np.zeros_like(x_normal)

        # Reconstruct full residual
        if single:
            full_residual = np.zeros(self.d)
            full_residual[self.outlier_idx] = out_residual if out_residual.ndim == 1 else out_residual[0]
            full_residual[self.normal_idx] = norm_residual if norm_residual.ndim == 1 else norm_residual[0]
        else:
            full_residual = np.zeros((batch, self.d))
            full_residual[:, self.outlier_idx] = out_residual
            full_residual[:, self.normal_idx] = norm_residual

        # QJL on full residual
        qjl_signs, residual_norms = self.qjl.quantize(full_residual if not single else full_residual)

        if single:
            return OutlierCompressedVector(
                outlier_indices=out_idx,
                outlier_norms=out_norms,
                normal_indices=norm_idx,
                normal_norms=norm_norms,
                qjl_signs=qjl_signs,
                residual_norms=residual_norms,
                effective_bits=self.effective_bits,
            )

        return OutlierCompressedVector(
            outlier_indices=out_idx,
            outlier_norms=out_norms,
            normal_indices=norm_idx,
            normal_norms=norm_norms,
            qjl_signs=qjl_signs,
            residual_norms=residual_norms,
            effective_bits=self.effective_bits,
        )

    def dequantize(self, compressed: OutlierCompressedVector) -> np.ndarray:
        """Dequantize outlier-strategy compressed vector."""
        single = compressed.qjl_signs.ndim == 1

        # Reconstruct outlier channels
        if self.pq_outlier is not None:
            x_outlier = self.pq_outlier.dequantize(compressed.outlier_indices, compressed.outlier_norms)
        else:
            x_outlier = np.zeros(0)

        # Reconstruct normal channels
        if self.pq_normal is not None:
            x_normal = self.pq_normal.dequantize(compressed.normal_indices, compressed.normal_norms)
        else:
            x_normal = np.zeros(0)

        # Reconstruct QJL residual
        x_qjl = self.qjl.dequantize(compressed.qjl_signs, compressed.residual_norms)

        # Combine
        if single:
            x_hat = np.zeros(self.d)
            if self.n_outlier > 0:
                x_hat[self.outlier_idx] = x_outlier
            if self.n_normal > 0:
                x_hat[self.normal_idx] = x_normal
            x_hat += x_qjl
        else:
            batch = compressed.qjl_signs.shape[0]
            x_hat = np.zeros((batch, self.d))
            if self.n_outlier > 0:
                x_hat[:, self.outlier_idx] = x_outlier
            if self.n_normal > 0:
                x_hat[:, self.normal_idx] = x_normal
            x_hat += x_qjl

        return x_hat

    def compression_ratio(self, original_bits: int = 16) -> float:
        """Compression ratio vs original precision."""
        # Effective bits per channel + norm overhead
        per_vector_bits = self.d * self.effective_bits + 32  # +32 for QJL norm
        # Also need outlier and normal norms: 2 × 32 bits
        per_vector_bits += 64
        original = self.d * original_bits
        return original / per_vector_bits
