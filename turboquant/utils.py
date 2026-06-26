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

"""Utility functions for bit packing and memory measurement."""

import numpy as np


def pack_bits(signs: np.ndarray) -> np.ndarray:
    """Pack {+1, -1} sign array into uint8 bitfield.

    8 signs per byte. +1 → 1, -1 → 0.

    Args:
        signs: int8 array of shape (d,) or (batch, d) with values {+1, -1}.

    Returns:
        uint8 array of shape (ceil(d/8),) or (batch, ceil(d/8)).
    """
    # Convert {+1, -1} → {1, 0}
    bits = (signs > 0).astype(np.uint8)

    if bits.ndim == 1:
        # Pad to multiple of 8
        padded_len = (len(bits) + 7) // 8 * 8
        padded = np.zeros(padded_len, dtype=np.uint8)
        padded[:len(bits)] = bits
        # Pack 8 bits into each byte
        packed = np.packbits(padded)
        return packed
    else:
        batch, d = bits.shape
        padded_len = (d + 7) // 8 * 8
        padded = np.zeros((batch, padded_len), dtype=np.uint8)
        padded[:, :d] = bits
        # packbits works on last axis
        packed = np.packbits(padded, axis=1)
        return packed


def unpack_bits(packed: np.ndarray, d: int) -> np.ndarray:
    """Unpack uint8 bitfield back to {+1, -1} signs.

    Args:
        packed: uint8 array from pack_bits.
        d: Original dimension (to truncate padding).

    Returns:
        int8 array of shape (d,) or (batch, d) with values {+1, -1}.
    """
    if packed.ndim == 1:
        bits = np.unpackbits(packed)[:d]
        # Convert {1, 0} → {+1, -1}
        return (bits.astype(np.int8) * 2 - 1)
    else:
        bits = np.unpackbits(packed, axis=1)[:, :d]
        return (bits.astype(np.int8) * 2 - 1)


def pack_indices(indices: np.ndarray, bit_width: int) -> np.ndarray:
    """Pack b-bit indices into compact byte array.

    For bit_width <= 4, packs multiple indices per byte.
    For bit_width <= 8, uses uint8 directly.

    Args:
        indices: Integer indices, shape (d,) or (batch, d).
        bit_width: Bits per index.

    Returns:
        Packed byte array.
    """
    if bit_width <= 0 or bit_width > 8:
        raise ValueError(f"bit_width must be 1-8, got {bit_width}")

    if bit_width <= 4:
        # For 1-4 bit, convert to binary then pack
        flat = indices.ravel().astype(np.uint8)
        # Convert each index to bit_width binary digits
        bits = np.zeros(len(flat) * bit_width, dtype=np.uint8)
        for b in range(bit_width):
            bits[b::bit_width] = (flat >> (bit_width - 1 - b)) & 1
        packed = np.packbits(bits)
        return packed.reshape(-1)  # flatten
    else:
        # 5-8 bit: just use uint8
        return indices.astype(np.uint8)


def memory_footprint_bytes(n_vectors: int, d: int, bit_width: int) -> dict:
    """Calculate memory footprint of compressed KV cache.

    Returns:
        Dict with breakdown: mse_indices, qjl_signs, norms, total, original_fp16.
    """
    mse_bits = bit_width - 1  # PolarQuant uses b-1 bits
    qjl_bits = 1

    mse_bytes = int(np.ceil(n_vectors * d * mse_bits / 8))
    qjl_bytes = int(np.ceil(n_vectors * d * qjl_bits / 8))
    norm_bytes = n_vectors * 4  # float32 per vector
    total = mse_bytes + qjl_bytes + norm_bytes
    original = n_vectors * d * 2  # fp16

    return {
        "mse_indices_bytes": mse_bytes,
        "qjl_signs_bytes": qjl_bytes,
        "norms_bytes": norm_bytes,
        "total_bytes": total,
        "original_fp16_bytes": original,
        "compression_ratio": original / total if total > 0 else float("inf"),
    }
