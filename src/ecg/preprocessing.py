"""
preprocessing.py

Signal-level preprocessing applied to every raw ECG before it reaches
the model:
  1. Bandpass filter (removes baseline wander + high-frequency noise)
  2. Per-lead z-score normalization
  3. Reduce the standard 12 leads down to the 8 independent leads
     (I, II, V1-V6) — III, aVR, aVL, aVF are linear combinations of
     I and II and carry no extra information for the model.

FIX LOG:
  - reduce_to_8_leads() now accepts BOTH 8-lead and 12-lead input.
    Previously it assumed 12 leads and crashed with
    "IndexError: index 8 is out of bounds for axis 1 with size 8"
    when the caller passed an already-reduced 8-lead signal.
"""

import numpy as np
from scipy.signal import butter, filtfilt

# Standard PTB-XL 12-lead order:
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

# The 8 independent leads we keep.
KEEP_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
KEEP_INDICES = [LEAD_ORDER.index(l) for l in KEEP_LEADS]


def bandpass_filter(
    signal: np.ndarray,
    fs: int = 100,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 3,
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter along the time axis.

    Args:
        signal: (n_samples, n_leads)
        fs: sampling rate in Hz
        lowcut / highcut: passband edges in Hz. 0.5-40Hz is standard for
            removing baseline wander and muscle/powerline noise while
            keeping clinically relevant QRS/ST morphology.

    Returns:
        Filtered signal, same shape as input.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = min(highcut / nyquist, 0.99)
    b, a = butter(order, [low, high], btype="band")
    # filtfilt is zero-phase (no time-shift), applied per-lead
    filtered = np.zeros_like(signal)
    for lead in range(signal.shape[1]):
        filtered[:, lead] = filtfilt(b, a, signal[:, lead])
    return filtered.astype(np.float32)


def normalize(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-lead z-score normalization: (x - mean) / std, computed
    independently for each lead so leads with different natural
    amplitudes are put on a comparable scale.
    """
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    return (signal - mean) / (std + eps)


def reduce_to_8_leads(signal: np.ndarray) -> np.ndarray:
    """
    Drop III, aVR, aVL, aVF — keep I, II, V1-V6.

    Handles two cases:
      - 12-lead input → reduce to 8 via KEEP_INDICES
      - 8-lead input  → return as-is (already reduced)

    This makes the function idempotent: calling it twice is safe,
    and it will not crash if an upstream step has already reduced
    the signal from 12 leads to 8.

    Args:
        signal: (n_samples, 12) OR (n_samples, 8)

    Returns:
        (n_samples, 8)

    Raises:
        ValueError: if the signal has any other number of leads.
    """
    n_leads = signal.shape[1]

    # Case 1: already reduced to 8 leads — no-op
    if n_leads == 8:
        return signal

    # Case 2: full 12-lead input — reduce using KEEP_INDICES
    if n_leads == 12:
        return signal[:, KEEP_INDICES]

    # Anything else is a bug upstream
    raise ValueError(
        f"reduce_to_8_leads expected 8 or 12 leads, got {n_leads}. "
        f"Check the shape of the signal entering preprocess_signal()."
    )


def preprocess_signal(
    signal: np.ndarray, fs: int = 100
) -> np.ndarray:
    """
    Full pipeline used everywhere else in the codebase: filter ->
    reduce leads -> normalize.

    Accepts either 12-lead or 8-lead input:
      - 12-lead → filtered → reduced to 8 → normalized
      - 8-lead  → filtered → returned as-is (already reduced) → normalized

    Args:
        signal: raw (n_samples, 12) OR (n_samples, 8)
        fs: sampling rate in Hz

    Returns:
        (n_samples, 8), filtered and normalized, ready for the model.
    """
    filtered = bandpass_filter(signal, fs=fs)
    reduced = reduce_to_8_leads(filtered)
    normalized = normalize(reduced)
    return normalized


def preprocess_batch(signals: np.ndarray, fs: int = 100) -> np.ndarray:
    """
    Apply preprocess_signal to a batch of shape (n, n_samples, 12)
    or (n, n_samples, 8). Returns (n, n_samples, 8).
    """
    return np.stack([preprocess_signal(s, fs=fs) for s in signals], axis=0)