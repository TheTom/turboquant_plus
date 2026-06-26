# Copyright 2026 Tom Turney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PolarQuant: Random rotation + optimal scalar quantization.

Algorithm 1 from the TurboQuant paper (AISTATS 2026).

After random rotation, coordinates follow a known Beta distribution (Gaussian in
high d), enabling optimal scalar quantization per coordinate independently.

Important: codebook is calibrated for unit-norm vectors. For non-unit-norm inputs,
we extract norms, normalize, quantize, then rescale on dequantization.
(Paper page 5: "store the L2 norms in floating-point precision and rescale")
"""

import numpy as np

from turboquant.codebook import optimal_centroids, nearest_centroid_indices
from turboquant.rotation import random_rotation_dense


class PolarQuant:
    """MSE-optimized vector quantizer via random rotation + scalar quantization.

    Handles arbitrary-norm vectors by extracting norms before quantization
    and rescaling after dequantization. This is critical for real KV cache
    tensors which are NOT unit-norm.

    Usage:
        pq = PolarQuant(d=128, bit_width=2, seed=42)
        indices, norms = pq.quantize(x)       # x: (d,) or (batch, d)
        x_hat = pq.dequantize(indices, norms)  # reconstructed
    """

    def __init__(self, d: int, bit_width: int, seed: int = 42, norm_correction: bool = True):
        self.d = d
        self.bit_width = bit_width
        self.n_centroids = 1 << bit_width
        self.norm_correction = norm_correction

        rng = np.random.default_rng(seed)
        self.rotation = random_rotation_dense(d, rng)
        self.centroids = optimal_centroids(bit_width, d)

    def quantize(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Quantize a vector or batch of vectors.

        Args:
            x: Input vector(s), shape (d,) or (batch, d).

        Returns:
            (indices, norms) where:
                indices: integer indices, shape (d,) or (batch, d)
                norms: L2 norms, scalar or (batch,) — needed for dequantization
        """
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]

        # Extract norms and normalize (paper page 5)
        norms = np.linalg.norm(x, axis=1)  # (batch,)
        # Avoid division by zero for zero vectors
        safe_norms = np.where(norms > 0, norms, 1.0)
        x_normalized = x / safe_norms[:, np.newaxis]

        # Rotate normalized vectors
        y = (self.rotation @ x_normalized.T).T

        # Nearest centroid per coordinate
        indices = nearest_centroid_indices(y, self.centroids)

        if single:
            return indices[0], norms[0]
        return indices, norms

    def dequantize(self, indices: np.ndarray, norms: np.ndarray) -> np.ndarray:
        """Dequantize indices back to vectors.

        Args:
            indices: Integer indices, shape (d,) or (batch, d).
            norms: Original L2 norms, scalar or (batch,).

        Returns:
            Reconstructed vectors, same shape as original input.
        """
        single = indices.ndim == 1
        if single:
            indices = indices[np.newaxis, :]
            norms = np.array([norms])

        # Look up centroids in the rotated domain.
        y_hat = self.centroids[indices]

        # Production TurboQuant rescales the reconstructed rotated vector by
        # grp_norm / recon_norm before the inverse rotation. In the normalized
        # Python prototype, grp_norm = 1, so this simplifies to re-normalizing
        # y_hat back to unit norm.
        if self.norm_correction:
            y_hat_norms = np.linalg.norm(y_hat, axis=1, keepdims=True)
            y_hat_norms = np.where(y_hat_norms > 1e-10, y_hat_norms, 1.0)
            y_hat = y_hat / y_hat_norms

        x_hat_unit = (self.rotation.T @ y_hat.T).T

        # Rescale by original norms
        x_hat = x_hat_unit * norms[:, np.newaxis]

        return x_hat[0] if single else x_hat

    def quantize_and_residual(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantize and return indices, norms, and residual error.

        Used by TurboQuant's second stage (QJL on residual).

        Returns:
            (indices, norms, residual) where residual = x - dequantize(indices, norms).
        """
        indices, norms = self.quantize(x)
        x_hat = self.dequantize(indices, norms)
        residual = x - x_hat
        return indices, norms, residual
