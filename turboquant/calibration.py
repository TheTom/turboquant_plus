import numpy as np

def calibrate_outlier_channels(vectors: np.ndarray, n_outliers: int) -> np.ndarray:
    """Identify outlier channels based on mean absolute magnitude.

    Args:
        vectors: Array of shape (seq_len, head_dim) containing the calibration data.
        n_outliers: Number of outlier channels to identify.

    Returns:
        1D array of shape (n_outliers,) containing the sorted indices of the most active channels.
    """
    if n_outliers == 0:
        return np.array([], dtype=int)
    
    # Compute mean absolute magnitude for each channel across all tokens
    mean_abs_mags = np.mean(np.abs(vectors), axis=0)
    
    # Get indices of channels with the highest magnitudes
    # argsort sorts ascending, so we take the last n_outliers and reverse to get descending
    sorted_indices = np.argsort(mean_abs_mags)
    outlier_idx = sorted_indices[-n_outliers:][::-1]
    
    return outlier_idx
