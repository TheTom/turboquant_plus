"""RotorQuant: Clifford algebra reimagining of TurboQuant.

Replaces the d×d random orthogonal matrix with Cl(3,0) rotors.
44× fewer parameters, matching attention fidelity on real models.

Compatible with the turboquant_plus codebase API.
"""

import numpy as np
from dataclasses import dataclass

from turboquant.clifford import (
    make_random_rotor, rotor_sandwich, embed_vectors, extract_vectors,
    gp_rotor_mv, MV_DIM,
)
from turboquant.codebook import optimal_centroids, nearest_centroid_indices
from turboquant.qjl import QJL


@dataclass
class RotorCompressedVector:
    """Container for a RotorQuant-compressed vector."""
    grade_indices: dict          # {grade_name: np.ndarray of indices}
    vector_norms: np.ndarray     # original ||x||_2
    qjl_signs: np.ndarray        # QJL sign bits
    residual_norms: np.ndarray   # ||residual||_2
    bit_width: int


class RotorQuant:
    """Full RotorQuant: Rotor decorrelation + grade-aware Lloyd-Max + QJL.

    Usage:
        rq = RotorQuant(d=128, bit_width=3, seed=42)
        compressed = rq.quantize(x)
        x_hat = rq.dequantize(compressed)
    """

    def __init__(self, d: int, bit_width: int, seed: int = 42):
        if bit_width < 2:
            raise ValueError("RotorQuant requires bit_width >= 2")

        self.d = d
        self.bit_width = bit_width
        self.mse_bits = bit_width - 1
        self.n_groups = (d + 2) // 3

        rng = np.random.default_rng(seed)

        # Per-group rotors — only store sparse components [s, b12, b13, b23]
        self.rotors_s = np.empty(self.n_groups)
        self.rotors_b12 = np.empty(self.n_groups)
        self.rotors_b13 = np.empty(self.n_groups)
        self.rotors_b23 = np.empty(self.n_groups)

        for g in range(self.n_groups):
            r = make_random_rotor(rng)
            self.rotors_s[g] = r[0]
            self.rotors_b12[g] = r[4]
            self.rotors_b13[g] = r[5]
            self.rotors_b23[g] = r[6]

        # Grade-aware codebooks
        d_eff = max(self.n_groups * MV_DIM, 64)
        self.centroids = {
            'scalar':   optimal_centroids(self.mse_bits, d_eff),
            'vector':   optimal_centroids(self.mse_bits, d_eff),
            'bivector': optimal_centroids(self.mse_bits, d_eff),
            'trivector': optimal_centroids(max(self.mse_bits - 1, 1), d_eff),
        }
        self.grade_map = {
            'scalar':   [0],
            'vector':   [1, 2, 3],
            'bivector': [4, 5, 6],
            'trivector': [7],
        }

        # QJL for residual correction
        self.qjl = QJL(d, seed=seed + 1000)

    def _apply_rotors(self, mv: np.ndarray) -> np.ndarray:
        """Apply per-group rotor sandwich. mv shape (batch, n_groups, 8)."""
        result = np.empty_like(mv)
        for g in range(self.n_groups):
            s, p12, p13, p23 = self.rotors_s[g], self.rotors_b12[g], self.rotors_b13[g], self.rotors_b23[g]
            result[:, g] = rotor_sandwich(s, p12, p13, p23, mv[:, g])
        return result

    def _unapply_rotors(self, mv: np.ndarray) -> np.ndarray:
        """Inverse rotor sandwich (negate bivectors)."""
        result = np.empty_like(mv)
        for g in range(self.n_groups):
            s, p12, p13, p23 = self.rotors_s[g], self.rotors_b12[g], self.rotors_b13[g], self.rotors_b23[g]
            result[:, g] = rotor_sandwich(s, -p12, -p13, -p23, mv[:, g])
        return result

    def _quantize_mv(self, mv_rot: np.ndarray) -> tuple[np.ndarray, dict]:
        """Grade-aware quantization on rotated multivectors."""
        mv_q = np.empty_like(mv_rot)
        all_indices = {}
        for grade_name, comp_idx in self.grade_map.items():
            centroids = self.centroids[grade_name]
            data = mv_rot[..., comp_idx]  # (batch, n_groups, n_comps)
            flat = data.reshape(data.shape[0], -1)
            idx = nearest_centroid_indices(flat, centroids)
            q_vals = centroids[idx]
            mv_q[..., comp_idx] = q_vals.reshape(data.shape)
            all_indices[grade_name] = idx
        return mv_q, all_indices

    def quantize(self, x: np.ndarray) -> RotorCompressedVector:
        """Quantize vector(s). x shape (d,) or (batch, d)."""
        single = x.ndim == 1
        if single:
            x = x[np.newaxis]

        # Normalize
        norms = np.linalg.norm(x, axis=-1)
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        x_unit = x / safe_norms[:, np.newaxis]

        # Embed → rotor → quantize → un-rotor → extract
        mv, orig_d = embed_vectors(x_unit)
        mv_rot = self._apply_rotors(mv)
        mv_q, grade_indices = self._quantize_mv(mv_rot)
        mv_recon = self._unapply_rotors(mv_q)
        x_hat_unit = extract_vectors(mv_recon, orig_d)
        x_hat = x_hat_unit * safe_norms[:, np.newaxis]

        # Residual for QJL
        residual = x - x_hat
        qjl_signs, residual_norms = self.qjl.quantize(residual)

        if single:
            norms = norms[0]

        return RotorCompressedVector(
            grade_indices=grade_indices,
            vector_norms=norms,
            qjl_signs=qjl_signs,
            residual_norms=residual_norms,
            bit_width=self.bit_width,
        )

    def dequantize(self, compressed: RotorCompressedVector) -> np.ndarray:
        """Reconstruct from compressed representation."""
        norms = compressed.vector_norms
        single = norms.ndim == 0
        if single:
            norms = norms[np.newaxis]

        # Reconstruct multivector from indices
        batch = compressed.grade_indices['scalar'].shape[0]
        mv_q = np.zeros((batch, self.n_groups, MV_DIM))
        for grade_name, comp_idx in self.grade_map.items():
            centroids = self.centroids[grade_name]
            idx = compressed.grade_indices[grade_name]
            vals = centroids[idx].reshape(batch, self.n_groups, len(comp_idx))
            mv_q[..., comp_idx] = vals

        mv_recon = self._unapply_rotors(mv_q)
        x_unit = extract_vectors(mv_recon, self.d)
        x_mse = x_unit * norms[:, np.newaxis]

        # QJL correction
        x_qjl = self.qjl.dequantize(compressed.qjl_signs, compressed.residual_norms)
        result = x_mse + x_qjl

        if single:
            result = result[0]
        return result

    @property
    def n_parameters(self) -> int:
        """Total parameters (rotor components + codebook centroids)."""
        rotor_params = self.n_groups * 4
        codebook_params = sum(len(c) for c in self.centroids.values())
        return rotor_params + codebook_params


class RotorQuantMSE:
    """MSE-only RotorQuant (no QJL). Use for V cache compression."""

    def __init__(self, d: int, bit_width: int, seed: int = 42):
        self.d = d
        self.bit_width = bit_width
        self.n_groups = (d + 2) // 3
        rng = np.random.default_rng(seed)

        self.rotors_s = np.empty(self.n_groups)
        self.rotors_b12 = np.empty(self.n_groups)
        self.rotors_b13 = np.empty(self.n_groups)
        self.rotors_b23 = np.empty(self.n_groups)

        for g in range(self.n_groups):
            r = make_random_rotor(rng)
            self.rotors_s[g] = r[0]
            self.rotors_b12[g] = r[4]
            self.rotors_b13[g] = r[5]
            self.rotors_b23[g] = r[6]

        d_eff = max(self.n_groups * MV_DIM, 64)
        self.centroids = optimal_centroids(bit_width, d_eff)

    def quantize(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (indices, norms)."""
        single = x.ndim == 1
        if single:
            x = x[np.newaxis]
        norms = np.linalg.norm(x, axis=-1)
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        x_unit = x / safe_norms[:, np.newaxis]

        mv, orig_d = embed_vectors(x_unit)
        # Apply rotors
        for g in range(self.n_groups):
            mv[:, g] = rotor_sandwich(
                self.rotors_s[g], self.rotors_b12[g],
                self.rotors_b13[g], self.rotors_b23[g], mv[:, g])
        # Quantize all components uniformly
        flat = mv.reshape(mv.shape[0], -1)
        idx = nearest_centroid_indices(flat, self.centroids)
        if single:
            return idx[0], norms[0]
        return idx, norms

    def dequantize(self, indices: np.ndarray, norms: np.ndarray) -> np.ndarray:
        single = indices.ndim == 1
        if single:
            indices = indices[np.newaxis]
            norms = np.array([norms])
        vals = self.centroids[indices]
        mv = vals.reshape(vals.shape[0], self.n_groups, MV_DIM)
        # Inverse rotors
        for g in range(self.n_groups):
            mv[:, g] = rotor_sandwich(
                self.rotors_s[g], -self.rotors_b12[g],
                -self.rotors_b13[g], -self.rotors_b23[g], mv[:, g])
        x = extract_vectors(mv, self.d)
        x = x * norms[:, np.newaxis]
        if single:
            return x[0]
        return x
