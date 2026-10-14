
"""
Grad-CAM visualization for an 8-lead ECG.

Expected signal shape: (1000, 8)
Expected Grad-CAM shape: (1000,)

Important:
The current model produces one shared temporal Grad-CAM map.
This function does not calculate separate explanations per lead.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


LEAD_NAMES = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]


def create_gradcam_figure(
    signal,
    gradcam_map,
    sampling_rate=100,
):
    """
    Create a waveform with a Grad-CAM colour overlay.

    Parameters
    ----------
    signal : array-like
        ECG data with shape (samples, 8) or (8, samples).

    gradcam_map : array-like
        One-dimensional temporal Grad-CAM map.

    sampling_rate : int
        ECG sampling frequency in Hz.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing all eight ECG leads and the heatmap.
    """

    signal = np.asarray(signal, dtype=np.float64)
    cam = np.asarray(gradcam_map, dtype=np.float64).squeeze()

    # Check ECG dimensions.
    if signal.ndim != 2:
        raise ValueError(
            f"ECG must be a 2D array; received {signal.shape}."
        )

    # Convert (8, samples) to (samples, 8) if necessary.
    if signal.shape[1] != 8 and signal.shape[0] == 8:
        signal = signal.T

    if signal.shape[1] != 8:
        raise ValueError(
            f"Expected 8 ECG leads; received {signal.shape}."
        )

    if cam.ndim != 1:
        raise ValueError(
            f"Grad-CAM must be one-dimensional; received {cam.shape}."
        )

    if len(signal) != len(cam):
        raise ValueError(
            f"ECG has {len(signal)} samples but Grad-CAM has "
            f"{len(cam)} values."
        )

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")

    if not np.isfinite(signal).all():
        raise ValueError("ECG contains NaN or infinite values.")

    if not np.isfinite(cam).all():
        raise ValueError("Grad-CAM contains NaN or infinite values.")

    # Normalize Grad-CAM to [0, 1] for display.
    cam_min = float(cam.min())
    cam_max = float(cam.max())

    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    time = np.arange(len(signal)) / float(sampling_rate)

    # Use the same colour scale for all leads.
    cmap = plt.get_cmap("jet")
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(
        nrows=8,
        ncols=1,
        figsize=(15, 11),
        sharex=True,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.88,
        top=0.94,
        bottom=0.07,
        hspace=0.18,
    )

    for index, ax in enumerate(axes):
        waveform = signal[:, index]

        # Create a small line segment between consecutive samples.
        points = np.column_stack((time, waveform))
        segments = np.stack(
            [points[:-1], points[1:]],
            axis=1,
        )

        # Colour each segment using its temporal Grad-CAM value.
        line = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidth=0.85,
        )

        line.set_array((cam[:-1] + cam[1:]) / 2.0)
        ax.add_collection(line)

        # Keep the complete waveform visible.
        low = float(np.min(waveform))
        high = float(np.max(waveform))
        amplitude_range = high - low

        if amplitude_range < 1e-8:
            margin = 1.0
        else:
            margin = amplitude_range * 0.15

        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(low - margin, high + margin)

        ax.set_ylabel(
            LEAD_NAMES[index],
            rotation=0,
            labelpad=20,
            va="center",
        )

        # Light grid, similar to a clean ECG report.
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.tick_params(axis="both", labelsize=8)

    axes[-1].set_xlabel("Time (seconds)")
    axes[-1].set_xticks(
        np.arange(0, time[-1] + 0.001, 1.0)
    )

    fig.suptitle(
        "ECG Waveforms with Grad-CAM Overlay",
        fontsize=14,
        fontweight="bold",
    )

    # Colour bar explains relative importance.
    scalar_map = ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])

    colour_bar = fig.colorbar(
        scalar_map,
        ax=list(axes),
        location="right",
        fraction=0.025,
        pad=0.02,
    )

    colour_bar.set_label(
        "Relative Grad-CAM importance",
        fontsize=9,
    )
    colour_bar.set_ticks([0.0, 0.5, 1.0])
    colour_bar.set_ticklabels(
        ["Lower", "Medium", "Higher"]
    )

    return fig
