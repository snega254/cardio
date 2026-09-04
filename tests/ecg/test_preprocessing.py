"""
test_preprocessing.py

Unit tests for preprocessing.py using synthetic signals — do NOT
require the actual PTB-XL dataset to be downloaded, so these run
fast in CI / on any machine.
"""

import numpy as np
import pytest

from src.ecg.preprocessing import (
    bandpass_filter,
    normalize,
    preprocess_signal,
    reduce_to_8_leads,
)


def make_fake_12_lead_signal(n_samples=1000, fs=100):
    """Synthetic 12-lead signal: sine wave + noise, different per lead."""
    t = np.linspace(0, n_samples / fs, n_samples)
    signal = np.zeros((n_samples, 12), dtype=np.float32)
    rng = np.random.default_rng(42)
    for lead in range(12):
        freq = 1.0 + 0.1 * lead
        signal[:, lead] = np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(n_samples)
    return signal


def test_bandpass_filter_shape_preserved():
    signal = make_fake_12_lead_signal()
    filtered = bandpass_filter(signal, fs=100)
    assert filtered.shape == signal.shape


def test_bandpass_filter_removes_dc_offset():
    signal = make_fake_12_lead_signal() + 5.0  # add DC offset
    filtered = bandpass_filter(signal, fs=100)
    # after a 0.5Hz highpass, the mean should be pulled close to 0
    assert abs(filtered[:, 0].mean()) < 1.0


def test_reduce_to_8_leads_shape():
    signal = make_fake_12_lead_signal()
    reduced = reduce_to_8_leads(signal)
    assert reduced.shape == (signal.shape[0], 8)


def test_normalize_zero_mean_unit_std():
    signal = make_fake_12_lead_signal()
    normalized = normalize(signal)
    assert np.allclose(normalized.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(normalized.std(axis=0), 1, atol=1e-2)


def test_preprocess_signal_full_pipeline():
    signal = make_fake_12_lead_signal()
    processed = preprocess_signal(signal, fs=100)
    assert processed.shape == (signal.shape[0], 8)
    assert not np.isnan(processed).any()
