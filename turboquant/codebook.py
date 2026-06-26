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

"""Codebook construction for PolarQuant.

After random rotation, each coordinate follows Beta(d/2, d/2) on [-1/√d, 1/√d],
which converges to N(0, 1/d) for large d. We use optimal scalar quantizers for this
distribution.

Paper provides closed-form centroids for 1-bit and 2-bit. For higher bit-widths,
we use Lloyd's algorithm on the Gaussian approximation.
"""

import numpy as np
from scipy import stats


def optimal_centroids(bit_width: int, d: int) -> np.ndarray:
    """Compute optimal MSE centroids for the post-rotation coordinate distribution.

    Args:
        bit_width: Number of bits per coordinate (1, 2, 3, 4, ...).
        d: Vector dimension (affects centroid scale).

    Returns:
        Sorted array of 2^bit_width centroids.
    """
    n_centroids = 1 << bit_width

    if bit_width == 1:
        c = np.sqrt(2.0 / (np.pi * d))
        return np.array([-c, c])

    if bit_width == 2:
        return np.array([-1.51, -0.453, 0.453, 1.51]) / np.sqrt(d)

    # For b >= 3, use Lloyd's algorithm on N(0, 1/d)
    return _lloyds_gaussian(n_centroids, sigma=1.0 / np.sqrt(d))


def _lloyds_gaussian(n_centroids: int, sigma: float, n_iter: int = 100) -> np.ndarray:
    """Lloyd's algorithm (iterative k-means) for optimal scalar quantization of N(0, sigma²).

    Args:
        n_centroids: Number of quantization levels (2^b).
        sigma: Standard deviation of the Gaussian.
        n_iter: Number of Lloyd iterations.

    Returns:
        Sorted array of optimal centroids.
    """
    # Initialize boundary positions from uniform quantiles
    boundaries = stats.norm.ppf(
        np.linspace(0, 1, n_centroids + 1)[1:-1], scale=sigma
    )
    centroids = np.zeros(n_centroids)

    # Initial centroids: conditional expectations within each region
    centroids[0] = _gaussian_conditional_expectation(sigma, -np.inf, boundaries[0])
    for i in range(1, n_centroids - 1):
        centroids[i] = _gaussian_conditional_expectation(sigma, boundaries[i - 1], boundaries[i])
    centroids[-1] = _gaussian_conditional_expectation(sigma, boundaries[-1], np.inf)

    for _ in range(n_iter):
        # Update boundaries (midpoints between consecutive centroids)
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # Update centroids (conditional expectations within each region)
        centroids[0] = _gaussian_conditional_expectation(sigma, -np.inf, boundaries[0])
        for i in range(1, n_centroids - 1):
            centroids[i] = _gaussian_conditional_expectation(sigma, boundaries[i - 1], boundaries[i])
        centroids[-1] = _gaussian_conditional_expectation(sigma, boundaries[-1], np.inf)

    return np.sort(centroids)


def _gaussian_conditional_expectation(sigma: float, a: float, b: float) -> float:
    """E[X | a < X < b] where X ~ N(0, sigma²).

    Uses the formula: E[X | a < X < b] = sigma² * (φ(a/σ) - φ(b/σ)) / (Φ(b/σ) - Φ(a/σ))
    where φ is the PDF and Φ is the CDF of standard normal.
    """
    a_std = a / sigma if np.isfinite(a) else a
    b_std = b / sigma if np.isfinite(b) else b

    # Use sf() for upper tail to avoid CDF cancellation at extreme values
    # prob = P(a < X/σ < b) using the more numerically stable formulation
    if not np.isfinite(a_std):
        prob = stats.norm.cdf(b_std)
    elif not np.isfinite(b_std):
        prob = stats.norm.sf(a_std)
    else:
        prob = stats.norm.cdf(b_std) - stats.norm.cdf(a_std)

    if prob < 1e-15:
        # For semi-infinite intervals, use asymptotic approximation
        if np.isfinite(a) and not np.isfinite(b):
            return a + sigma  # E[X | X > a] ≈ a + σ for extreme a
        elif not np.isfinite(a) and np.isfinite(b):
            return b - sigma
        elif np.isfinite(a) and np.isfinite(b):
            return (a + b) / 2.0
        else:  # pragma: no cover — both infinite always has prob=1
            return 0.0

    pdf_diff = stats.norm.pdf(a_std) - stats.norm.pdf(b_std)
    return sigma * pdf_diff / prob


def nearest_centroid_indices(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Find nearest centroid index for each value. Vectorized.

    Args:
        values: Array of values to quantize, shape (...).
        centroids: Sorted centroid array, shape (n_centroids,).

    Returns:
        Integer indices into centroids array, same shape as values.
    """
    # Use searchsorted for sorted centroids — O(n log k) instead of O(n * k)
    # Find the insertion point, then check left and right neighbors
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return np.searchsorted(boundaries, values.ravel()).reshape(values.shape)
