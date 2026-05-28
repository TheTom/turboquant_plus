"""QJL: Quantized Johnson-Lindenstrauss Transform.

1-bit quantization via random orthogonal projection → sign to compress
vectors while preserving inner products. Stores a full d×d projection
matrix (O(d²) memory); for large d a structured/seeded approach would be
needed.

Key properties (orthogonal S):
    Q_qjl(x) = sign(S · x), S orthogonal
    Q_qjl_inv(z) = √(π/2) / √d · ||x|| · S^T · z
    E[⟨x̂, y⟩] = ⟨x, y⟩  (unbiased, paper Theorem 2)
    E[||x̂||²] = (π/2) · ||x||²
"""

import numpy as np


QJL_CONST = np.sqrt(np.pi / 2)


class QJL:
    """Quantized Johnson-Lindenstrauss 1-bit quantizer.

    Uses a random orthogonal projection matrix for minimum variance in the
    inner-product estimation. Orthogonality is enforced in __init__ and is
    a required invariant for the classical unbiased estimator.

    Usage:
        qjl = QJL(d=128, seed=42)
        signs, norm = qjl.quantize(residual)
        r_hat = qjl.dequantize(signs, norm)
    """

    _ORTHO_TOL = 1e-10

    def __init__(self, d: int, seed: int = 123):
        """
        Args:
            d: Vector dimension.
            seed: Random seed for projection matrix.
        """
        self.d = d
        rng = np.random.default_rng(seed)
        G = rng.standard_normal((d, d))
        Q, R = np.linalg.qr(G)
        diag_signs = np.sign(np.diag(R))
        Q = Q * diag_signs[np.newaxis, :]
        self.S = Q

        # Orthogonality contract: ||S Sᵀ − I||_F must be ~0.
        # Required for E[⟨x̂, y⟩] = ⟨x, y⟩ and E[||x̂||²] = (π/2)·||x||².
        ortho_err = np.linalg.norm(self.S @ self.S.T - np.eye(d), "fro")
        assert ortho_err < self._ORTHO_TOL, (
            f"QJL projection matrix not orthogonal: ||S Sᵀ − I||_F = {ortho_err:.2e} "
            f"(tolerance {self._ORTHO_TOL:.0e})"
        )

    def quantize(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Quantize residual vector(s) to sign bits.

        Args:
            r: Residual vector(s), shape (d,) or (batch, d).

        Returns:
            (signs, norms) where:
                signs: {+1, -1}^d or (batch, d), stored as int8
                norms: scalar or (batch,) — ||r||_2, needed for dequantization
        """
        single = r.ndim == 1
        if single:
            r = r[np.newaxis, :]

        norms = np.linalg.norm(r, axis=1)
        projected = (self.S @ r.T).T
        signs = np.sign(projected).astype(np.int8)
        signs[signs == 0] = 1

        if single:
            return signs[0], norms[0]
        return signs, norms

    def dequantize(
        self,
        signs: np.ndarray,
        norms: np.ndarray,
        shrinkage: float = 1.0,
    ) -> np.ndarray:
        """Dequantize sign bits back to approximate residual.

        Args:
            signs: Sign bits, shape (d,) or (batch, d).
            norms: Residual norms, scalar or (batch,).
            shrinkage: Multiplicative factor applied to the classical QJL
                reconstruction. The default of ``1.0`` is the paper's
                classical unbiased estimator: E[⟨x̂, y⟩] = ⟨x, y⟩.

                The MMSE-optimal value is ``2/np.pi ≈ 0.6366``. Derivation:
                with orthogonal S and the √d scale, E[||x̂||²] = (π/2)·||x||²
                and E[⟨x̂, x⟩] = ||x||², so the α minimising
                E[||αx̂ − x||²] = α²·(π/2)·||x||² − 2α·||x||² + ||x||² is
                α* = 2/π. Shrinking by α* gives a biased but lower-MSE
                estimator that is preferable for downstream MSE-sensitive
                consumers (e.g. attention-score correction).

        Returns:
            Approximate residual, same shape as original.
        """
        single = signs.ndim == 1
        if single:
            signs = signs[np.newaxis, :].astype(np.float64)
            norms = np.array([norms])
        else:
            signs = signs.astype(np.float64)

        # x̂ = √(π/2) / √d · ||x|| · S^T · signs
        reconstructed = (self.S.T @ signs.T).T
        scale = QJL_CONST / np.sqrt(self.d) * norms
        reconstructed *= (scale[:, np.newaxis] * shrinkage)

        return reconstructed[0] if single else reconstructed
