"""
Signal conversion with light post-processing.
Preserves amplitude.
"""

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter


PAPER_SPEED_MM_PER_SEC = 25.0
GAIN_MM_PER_MV = 10.0
SIGNAL_SAMPLE_RATE_HZ = 100.0
MODEL_DURATION_SECONDS = 10.0


def pixels_to_seconds(x, small_box_px):
    return (x / small_box_px) / PAPER_SPEED_MM_PER_SEC


def pixels_to_millivolts(y, small_box_px, baseline_y):
    return ((baseline_y - y) / small_box_px) / GAIN_MM_PER_MV


def estimate_baseline(points):
    if len(points) == 0:
        return 0.0
    return float(np.median(points[:, 1]))


def remove_drift(signal, fs=100.0, cutoff=0.3):
    if len(signal) < 20:
        return signal
    nyq = fs / 2.0
    cutoff_n = max(0.01, cutoff / nyq)
    try:
        b, a = butter(2, cutoff_n, btype="high")
        return filtfilt(b, a, signal)
    except Exception:
        return signal


def points_to_signal(
    points,
    small_box_px,
    target_length=1000,
    duration_seconds=None,
):
    """Convert traced points to mV signal with light post-processing."""
    if len(points) < 2:
        return np.zeros(target_length, dtype=np.float32)

    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)

    if (y.max() - y.min()) < 3:
        return np.zeros(target_length, dtype=np.float32)

    order = np.argsort(x)
    x, y = x[order], y[order]

    ux, idx = np.unique(x, return_index=True)
    x, y = ux, y[idx]

    baseline_y = estimate_baseline(np.column_stack([x, y]))
    time_s = pixels_to_seconds(x, small_box_px)
    voltage_mv = pixels_to_millivolts(y, small_box_px, baseline_y)

    if time_s[-1] == time_s[0]:
        return np.zeros(target_length, dtype=np.float32)

    # Preserve the paper's real timing. The model window is fixed at ten
    # seconds, so shorter strips are padded and longer strips are cropped.
    duration_s = (
        max(0.0, float(duration_seconds))
        if duration_seconds is not None
        else max(0.0, float(time_s[-1] - time_s[0]))
    )
    native_length = max(2, int(round(duration_s * SIGNAL_SAMPLE_RATE_HZ)) + 1)
    if native_length <= target_length:
        target_end = time_s[0] + duration_s
        t_target = np.linspace(time_s[0], target_end, native_length)
        signal = np.interp(t_target, time_s, voltage_mv)
    else:
        t_target = np.linspace(time_s[0], time_s[0] + (target_length - 1) / SIGNAL_SAMPLE_RATE_HZ, target_length)
        signal = np.interp(t_target, time_s, voltage_mv)

    # Light post-processing
    signal = signal - signal.mean()
    signal = remove_drift(signal, fs=100.0, cutoff=0.3)

    if len(signal) > 15:
        signal = savgol_filter(signal, window_length=5, polyorder=2)

    # Pad only after image-derived samples have been filtered. Holding the
    # final value avoids a synthetic zero step at the image/padding boundary.
    if native_length < target_length:
        padded = np.full(target_length, signal[-1], dtype=float)
        padded[:native_length] = signal
        signal = padded

    return signal.astype(np.float32)