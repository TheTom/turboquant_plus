"""TurboQuant: Full algorithm combining PolarQuant + QJL.

Algorithm 2 from the paper — Inner Product TurboQuant.

Two-stage process:
1. PolarQuant at (b-1) bits for MSE-optimal compression
2. QJL at 1 bit on the residual for bias elimination

Total: b bits per coordinate with near-optimal inner product distortion.
"""

import numpy as np
from dataclasses import dataclass

from turboquant.polar_quant import PolarQuant
from turboquant.qjl import QJL


@dataclass
class CompressedVector:
    """Container for a TurboQuant-compressed vector."""
    mse_indices: np.ndarray    # (d,) or (batch, d) — PolarQuant indices, (b-1)-bit integers
    vector_norms: np.ndarray   # scalar or (batch,) — original ||x||_2 for rescaling
    qjl_signs: np.ndarray      # (d,) or (batch, d) — QJL sign bits, int8 {+1, -1}
    residual_norms: np.ndarray  # scalar or (batch,) — ||residual||_2
    bit_width: int              # total bits per coordinate


class TurboQuant:
    """Full TurboQuant quantizer: PolarQuant (b-1 bits) + QJL (1 bit).

    Usage:
        tq = TurboQuant(d=128, bit_width=3, seed=42)
        compressed = tq.quantize(x)
        x_hat = tq.dequantize(compressed)

        # Verify inner product preservation
        ip_original = np.dot(x, y)
        ip_approx = np.dot(tq.dequantize(tq.quantize(x)),
                           tq.dequantize(tq.quantize(y)))
    """

    def __init__(self, d: int, bit_width: int, seed: int = 42, norm_correction: bool = True):
        """
        Args:
            d: Vector dimension.
            bit_width: Total bits per coordinate (b). PolarQuant uses b-1, QJL uses 1.
            seed: Random seed for both rotation and projection matrices.
        """
        if bit_width < 2:
            raise ValueError("TurboQuant requires bit_width >= 2 (1 bit PolarQuant + 1 bit QJL). "
                             "For 1-bit, use QJL directly.")

        self.d = d
        self.bit_width = bit_width

        # Spawn independent child seeds from a SeedSequence so PolarQuant and QJL
        # use statistically independent random streams without magic offset arithmetic.
        # Accept either an int or an already-created SeedSequence (e.g. from a parent spawner).
        ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        pq_child, qjl_child = ss.spawn(2)

        # Stage 1: PolarQuant at (b-1) bits
        self.polar_quant = PolarQuant(
            d, bit_width=bit_width - 1, seed=pq_child, norm_correction=norm_correction,
        )

        # Stage 2: QJL for residual (independent seed stream)
        self.qjl = QJL(d, seed=qjl_child)

    def quantize(self, x: np.ndarray) -> CompressedVector:
        """Quantize a vector or batch.

        Args:
            x: Input vector(s), shape (d,) or (batch, d).

        Returns:
            CompressedVector containing indices, signs, and norms.
        """
        # Stage 1: PolarQuant (with norm extraction)
        mse_indices, vector_norms, residual = self.polar_quant.quantize_and_residual(x)

        # Stage 2: QJL on residual
        qjl_signs, residual_norms = self.qjl.quantize(residual)

        return CompressedVector(
            mse_indices=mse_indices,
            vector_norms=vector_norms,
            qjl_signs=qjl_signs,
            residual_norms=residual_norms,
            bit_width=self.bit_width,
        )

    def dequantize(self, compressed: CompressedVector) -> np.ndarray:
        """Dequantize back to approximate vector.

        Args:
            compressed: CompressedVector from quantize().

        Returns:
            Reconstructed vector(s), same shape as original.
        """
        # Stage 1: PolarQuant reconstruction (with norm rescaling)
        x_mse = self.polar_quant.dequantize(compressed.mse_indices, compressed.vector_norms)

        # Stage 2: QJL residual reconstruction
        x_qjl = self.qjl.dequantize(compressed.qjl_signs, compressed.residual_norms)

        return x_mse + x_qjl

    def compressed_size_bits(self, n_vectors: int) -> int:
        """Compute total storage in bits for n_vectors compressed vectors.

        Includes:
        - PolarQuant indices: (b-1) bits per coordinate per vector
        - QJL signs: 1 bit per coordinate per vector
        - Residual norms: 32 bits (float32) per vector
        """
        per_vector = self.d * self.bit_width  # (b-1) + 1 bits per coordinate
        norms = 32  # float32 per vector
        return n_vectors * (per_vector + norms)

    def compression_ratio(self, original_bits_per_value: int = 16) -> float:
        """Compute compression ratio vs original precision.

        Args:
            original_bits_per_value: Bits per value in the original cache (16 for fp16).

        Returns:
            Compression ratio (e.g., 4.0 means 4× smaller).
        """
        original_per_vector = self.d * original_bits_per_value
        compressed_per_vector = self.d * self.bit_width + 32  # +32 for norm
        return original_per_vector / compressed_per_vector


class TurboQuantMSE:
    """MSE-only TurboQuant (Algorithm 1) — no QJL stage.

    Use for V cache compression where MSE matters more than inner product.
    Simpler, slightly less storage overhead (no QJL signs needed).
    """

    def __init__(self, d: int, bit_width: int, seed: int = 42, norm_correction: bool = True):
        self.d = d
        self.bit_width = bit_width
        self.polar_quant = PolarQuant(
            d, bit_width=bit_width, seed=seed, norm_correction=norm_correction,
        )

    def quantize(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (indices, norms)."""
        return self.polar_quant.quantize(x)

    def dequantize(self, indices: np.ndarray, norms: np.ndarray) -> np.ndarray:
        return self.polar_quant.dequantize(indices, norms)

    def compressed_size_bits(self, n_vectors: int) -> int:
        """Compute total storage in bits for n_vectors compressed vectors.

        Includes:
        - PolarQuant indices: b bits per coordinate per vector
        - Norms: 32 bits (float32) per vector (stored for per-vector rescaling)
        """
        per_vector = self.d * self.bit_width
        norms = 32  # float32 per vector
        return n_vectors * (per_vector + norms)
