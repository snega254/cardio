"""
Band detector using adaptive projection.

The trace has VERY dark pixels (gray < 80). The grid does not.
So use the count of "very dark" pixels per row as the projection metric.
"""

import numpy as np
from typing import Optional

from src.ecg.layout_templates import LAYOUT_TEMPLATES


def _to_gray(image):
    if image.ndim == 3:
        return image.mean(axis=2)
    return image.astype(float)


def _find_runs_above(projection, threshold, min_run):
    above = projection > threshold
    runs = []
    start = None
    for i, val in enumerate(above):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_run:
                runs.append((start, i))
            start = None
    if start is not None and len(projection) - start >= min_run:
        runs.append((start, len(projection)))
    return runs


def _merge_close(runs, max_gap):
    if len(runs) <= 1:
        return runs
    merged = [runs[0]]
    for s, e in runs[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e <= max_gap:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))
    return merged


def detect_bands(image, dark_threshold=80, min_run_size=40, max_gap=80,
                 margin_pct=0.02):
    """
    Detect horizontal trace bands.

    Uses VERY dark threshold (default 80) to catch only the black trace.
    """
    gray = _to_gray(image)
    h, w = gray.shape

    top = int(h * margin_pct)
    left = int(w * margin_pct)
    right = w - left

    # Very dark pixels only (the trace)
    dark = gray < dark_threshold

    # Count very-dark pixels per row (only in main column area)
    row_proj = dark[:, left:right].sum(axis=1).astype(float)
    row_proj[:top] = 0

    # Threshold: at least 15 dark pixels in a row
    row_thresh = 15
    h_runs = _find_runs_above(row_proj, row_thresh, min_run_size)

    return {
        "horizontal_bands": h_runs,
        "vertical_bands": [],
        "n_rows": len(h_runs),
        "n_cols": 0,
    }


def split_columns(image, y0, y1, dark_threshold=80):
    """
    Within a given horizontal band, split into columns.
    Returns list of (x0, x1) column ranges.
    """
    gray = _to_gray(image)
    h, w = gray.shape

    if y1 > h:
        y1 = h

    band = gray[y0:y1, :]
    dark = band < dark_threshold

    # Column projection
    col_proj = dark.sum(axis=0).astype(float)

    # Threshold: at least 10% of band height dark pixels
    thresh = max(3, int((y1 - y0) * 0.05))
    runs = _find_runs_above(col_proj, thresh, min_run=60)
    runs = _merge_close(runs, max_gap=200)

    return runs


if __name__ == "__main__":
    from src.ecg.image_loader import load_ecg_image

    print("Loading Normal-3.jpg...")
    img = load_ecg_image("Normal-3.jpg")
    print(f"Image shape: {img.shape}\n")

    for thresh in [70, 80, 90, 100]:
        print(f"--- dark_threshold = {thresh} ---")
        info = detect_bands(img, dark_threshold=thresh)
        print(f"  Horizontal bands: {info['n_rows']}")
        for i, (y0, y1) in enumerate(info["horizontal_bands"]):
            print(f"    Row {i+1}: y = {y0:4d} to {y1:4d}  (h={y1-y0})")
        print()

    # For the best threshold, examine columns in each row
    print("=" * 70)
    print("Now examining columns within each detected row (thresh=80):")
    print("=" * 70)
    info = detect_bands(img, dark_threshold=80)
    for i, (y0, y1) in enumerate(info["horizontal_bands"]):
        cols = split_columns(img, y0, y1, dark_threshold=80)
        print(f"  Row {i+1} (y={y0}-{y1}): {len(cols)} column(s)")
        for j, (x0, x1) in enumerate(cols):
            print(f"    Col {j+1}: x = {x0:4d} to {x1:4d}  (w={x1-x0})")