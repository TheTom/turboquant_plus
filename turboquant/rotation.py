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

"""Random rotation matrix generation for PolarQuant.

Two implementations:
1. Dense Haar-distributed rotation via QR decomposition — O(d²) multiply, exact
2. Fast structured rotation via Hadamard + random sign flips — O(d log d), approximate
"""

import numpy as np


def random_rotation_dense(d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a Haar-distributed random rotation matrix via QR decomposition.

    Args:
        d: Dimension of the rotation matrix. Must be >= 1.
        rng: NumPy random generator for reproducibility.

    Returns:
        Orthogonal matrix Π ∈ R^(d×d) with det(Π) = +1.
    """
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    # Random Gaussian matrix
    G = rng.standard_normal((d, d))
    # QR decomposition gives orthogonal Q
    Q, R = np.linalg.qr(G)
    # Make Q Haar-distributed by fixing signs via diagonal of R
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs[np.newaxis, :]
    # Ensure proper rotation (det = +1) — flip first column if det = -1
    # Use slogdet for numerical stability (det overflows for large d)
    sign, _ = np.linalg.slogdet(Q)
    if sign < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def hadamard_matrix(n: int) -> np.ndarray:
    """Generate a normalized Hadamard matrix of size n (must be power of 2).

    Uses the recursive Sylvester construction.
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a positive power of 2, got {n}")
    if n == 1:
        return np.array([[1.0]])
    half = hadamard_matrix(n // 2)
    H = np.block([[half, half], [half, -half]])
    return H


def random_rotation_fast(d: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate a fast structured random rotation: D @ H @ D' (random signs + Hadamard).

    For large d, this is O(d log d) to apply instead of O(d²).
    Returns components separately for fast application.

    Args:
        d: Original dimension.
        rng: NumPy random generator.

    Returns:
        Tuple of (signs1, signs2, padded_d) where the rotation is applied as:
            1. Pad x to padded_d if needed
            2. x *= signs1
            3. x = H @ x / sqrt(padded_d)  (use fast Walsh-Hadamard)
            4. x *= signs2
            5. Truncate back to d
    """
    padded_d = _next_power_of_2(d)
    signs1 = rng.choice([-1.0, 1.0], size=padded_d)
    signs2 = rng.choice([-1.0, 1.0], size=padded_d)
    return signs1, signs2, padded_d


def fast_walsh_hadamard_transform(x: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform, O(n log n).

    Args:
        x: Input array of length n (must be power of 2). Not modified in-place.

    Returns:
        New transformed array (normalized by 1/sqrt(n)).
    """
    n = len(x)
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"Input length must be a positive power of 2, got {n}")
    x = x.copy().astype(np.float64)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = x[j]
                b = x[j + h]
                x[j] = a + b
                x[j + h] = a - b
        h *= 2
    return x / np.sqrt(n)


def apply_fast_rotation(x: np.ndarray, signs1: np.ndarray, signs2: np.ndarray, padded_d: int) -> np.ndarray:
    """Apply the structured random rotation to a vector.

    Args:
        x: Input vector of dimension d.
        signs1, signs2: Random sign vectors from random_rotation_fast.
        padded_d: Padded dimension (power of 2).

    Returns:
        Rotated vector of dimension d.
    """
    d = len(x)
    # Pad to power of 2
    padded = np.zeros(padded_d)
    padded[:d] = x
    # D1 @ x
    padded *= signs1
    # H @ D1 @ x (normalized)
    padded = fast_walsh_hadamard_transform(padded)
    # D2 @ H @ D1 @ x
    padded *= signs2
    return padded[:d]


def apply_fast_rotation_transpose(y: np.ndarray, signs1: np.ndarray, signs2: np.ndarray, padded_d: int) -> np.ndarray:
    """Apply the transpose of the structured random rotation.

    Since D and H are their own transposes (symmetric), the transpose is D1 @ H @ D2.
    """
    d = len(y)
    padded = np.zeros(padded_d)
    padded[:d] = y
    # Reverse order: D2^T = D2, H^T = H, D1^T = D1
    padded *= signs2
    padded = fast_walsh_hadamard_transform(padded)
    padded *= signs1
    return padded[:d]


def apply_fast_rotation_batch(X: np.ndarray, signs1: np.ndarray, signs2: np.ndarray, padded_d: int) -> np.ndarray:
    """Apply structured rotation to a batch of vectors. Shape: (batch, d)."""
    batch, d = X.shape
    padded = np.zeros((batch, padded_d))
    padded[:, :d] = X
    padded *= signs1[np.newaxis, :]

    # Vectorized Walsh-Hadamard on each row
    n = padded_d
    h = 1
    while h < n:
        # Reshape for butterfly operations
        reshaped = padded.reshape(batch, n // (h * 2), 2, h)
        a = reshaped[:, :, 0, :].copy()
        b = reshaped[:, :, 1, :].copy()
        reshaped[:, :, 0, :] = a + b
        reshaped[:, :, 1, :] = a - b
        padded = reshaped.reshape(batch, n)
        h *= 2

    padded /= np.sqrt(n)
    padded *= signs2[np.newaxis, :]
    return padded[:, :d]
