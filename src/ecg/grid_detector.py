"""
Detect ECG paper grid and compute pixel-to-mm calibration.
"""

import numpy as np
import cv2


def detect_grid(image: np.ndarray) -> dict:
    """
    Detect the ECG grid on a printout.

    Args:
        image: RGB image (H, W, 3)

    Returns:
        dict with:
            - small_box_px: float (avg pixels per 1mm box)
            - large_box_px: float (avg pixels per 5mm box)
            - is_red_grid: bool
            - grid_color: (R, G, B) tuple
            - confidence: float 0-1
    """
    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # ECG grid is usually red/pink or light orange
    # Red grid: hue near 0 or 180
    # Green grid: hue near 60
    # Gray grid: low saturation, medium value

    # Detect red grid
    lower_red1 = np.array([0, 30, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 30, 100])
    upper_red2 = np.array([180, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | \
               cv2.inRange(hsv, lower_red2, upper_red2)

    red_ratio = np.count_nonzero(mask_red) / mask_red.size

    # Detect grid via line projection on gray version
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Use adaptive threshold to find grid lines
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )

    # Horizontal projection (rows with many dark pixels = grid lines)
    h_proj = binary.sum(axis=1) / 255
    v_proj = binary.sum(axis=0) / 255

    # Find peaks in projections
    h_peaks = _find_periodic_peaks(h_proj)
    v_peaks = _find_periodic_peaks(v_proj)

    # Small box = distance between adjacent grid lines
    small_box_px = 0.0
    if len(h_peaks) > 5:
        h_diffs = np.diff(h_peaks)
        small_box_px = float(np.median(h_diffs[h_diffs > 2]))

    if len(v_peaks) > 5:
        v_diffs = np.diff(v_peaks)
        v_median = float(np.median(v_diffs[v_diffs > 2]))
        if small_box_px == 0:
            small_box_px = v_median
        else:
            small_box_px = (small_box_px + v_median) / 2

    # Fallback if grid not detected
    if small_box_px < 2 or small_box_px > 50:
        # Assume standard: 1mm grid ~ 3-5 pixels at typical resolutions
        small_box_px = 4.0

    return {
        "small_box_px": small_box_px,
        "large_box_px": small_box_px * 5,
        "is_red_grid": red_ratio > 0.05,
        "grid_color": _get_dominant_color(image, mask_red) if red_ratio > 0.05 else (0, 0, 0),
        "confidence": min(1.0, len(h_peaks) / 50),
    }


def _find_periodic_peaks(projection: np.ndarray, min_dist: int = 3) -> np.ndarray:
    """Find peaks in a 1D projection."""
    threshold = projection.mean() + projection.std()
    peaks = []
    last_peak = -min_dist
    for i, v in enumerate(projection):
        if v > threshold and (i - last_peak) >= min_dist:
            peaks.append(i)
            last_peak = i
    return np.array(peaks)


def _get_dominant_color(image: np.ndarray, mask: np.ndarray) -> tuple:
    """Get average RGB of masked region."""
    if mask.sum() == 0:
        return (0, 0, 0)
    pixels = image[mask > 0]
    return tuple(pixels.mean(axis=0).astype(int))


if __name__ == "__main__":
    import sys
    from src.ecg.image_loader import load_ecg_image
    if len(sys.argv) > 1:
        img = load_ecg_image(sys.argv[1])
        info = detect_grid(img)
        for k, v in info.items():
            print(f"{k}: {v}")
    else:
        print("Usage: python -m src.ecg.grid_detector <image_path>")