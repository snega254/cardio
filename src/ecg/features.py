"""Interpretable measurements calculated from a numerical ECG signal."""

from typing import Optional, Sequence

import numpy as np
from scipy.signal import find_peaks

DEFAULT_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]


def _lead_features(signal: np.ndarray, fs: float) -> dict:
    values = np.asarray(signal, dtype=np.float32)
    finite = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    std = float(finite.std())
    peaks, _ = find_peaks(
        finite,
        distance=max(1, int(0.25 * fs)),
        prominence=max(0.05, 0.5 * std),
    )
    duration_s = len(finite) / float(fs) if fs > 0 else 0.0
    heart_rate = float(len(peaks) * 60.0 / duration_s) if duration_s > 0 else 0.0
    return {
        "mean": float(finite.mean()),
        "std": std,
        "min": float(finite.min()) if len(finite) else 0.0,
        "max": float(finite.max()) if len(finite) else 0.0,
        "peak_to_peak": float(np.ptp(finite)) if len(finite) else 0.0,
        "rms": float(np.sqrt(np.mean(finite ** 2))) if len(finite) else 0.0,
        "detected_peak_count": int(len(peaks)),
        "estimated_heart_rate_bpm": heart_rate,
    }


def extract_ecg_features(
    signal: np.ndarray,
    lead_names: Optional[Sequence[str]] = None,
    fs: float = 100.0,
) -> dict:
    """Extract robust descriptive features from ``(n_samples, n_leads)`` signal."""
    values = np.asarray(signal)
    if values.ndim != 2:
        raise ValueError(f"ECG signal must have shape (n_samples, n_leads), got {values.shape}")
    names = list(lead_names or DEFAULT_LEADS)
    if len(names) != values.shape[1]:
        raise ValueError("lead_names length must match the signal's lead dimension")
    per_lead = {name: _lead_features(values[:, index], fs) for index, name in enumerate(names)}
    lead_heart_rates = [item["estimated_heart_rate_bpm"] for item in per_lead.values() if item["detected_peak_count"]]
    return {
        "sampling_rate_hz": float(fs),
        "n_samples": int(values.shape[0]),
        "n_leads": int(values.shape[1]),
        "leads": names,
        "per_lead": per_lead,
        "median_estimated_heart_rate_bpm": float(np.median(lead_heart_rates)) if lead_heart_rates else 0.0,
    }
