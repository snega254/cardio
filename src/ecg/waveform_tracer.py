"""
ECG waveform tracer using connected components + per-column continuity.

Strategy:
1. Threshold very-dark pixels — the trace is much darker than the grid
2. Remove thin horizontal AND vertical grid lines using narrow kernels
   that do NOT destroy wide horizontal trace segments (ST/T waves)
3. Do NOT keep only the largest component — keep all ink; per-column
   continuity picks the correct pixel
4. Trace per column with a generous continuity window
5. Smooth the traced y-values
6. Skip label remnant at the left edge and any calibration pulse at the start
"""

import numpy as np
import cv2
from scipy.signal import savgol_filter


def trace_waveform(image, region, debug=False):
    """
    Trace a single lead's waveform from a rectangular region.
    """
    x, y, w, h = region["x"], region["y"], region["width"], region["height"]

    # Clamp to image bounds
    H, W = image.shape[:2]
    x = max(0, x)
    y = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    crop = image[y:y2, x:x2].copy()

    if crop.size == 0:
        if debug:
            print("  Empty crop — region outside image")
        return np.empty((0, 2))

    # -------- Trim label remnant at left edge --------
    if crop.shape[1] > 16:
        crop = crop[:, 6:]
        w = crop.shape[1]

    # Remove colored paper grid before thresholding the black waveform.
    hsv_grid = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    red_grid = (
        cv2.inRange(hsv_grid, np.array([0, 30, 100]), np.array([10, 255, 255]))
        | cv2.inRange(hsv_grid, np.array([170, 30, 100]), np.array([180, 255, 255]))
    )
    crop[red_grid > 0] = 255

    # -------------------------------------------
    # Step 1: Trace mask (very dark, achromatic)
    # -------------------------------------------
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Trace is black: low V, low S
    trace_mask = ((V < 120) & (S < 85)).astype(np.uint8) * 255

    # Fallback: pure grayscale threshold
    if np.count_nonzero(trace_mask) < 50:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        trace_mask = (gray < 110).astype(np.uint8) * 255

    # Red grid pixels have already been removed above. Avoid subtracting
    # vertical structures here: that also erases the steep QRS complexes.
    cleaned = cv2.medianBlur(trace_mask, 3)

    if debug:
        n = np.count_nonzero(cleaned)
        print(f"  Mask: {n} px (crop {w}x{h})")

    # -------------------------------------------
    # Step 3: Trace per column with continuity
    # -------------------------------------------
    points = _trace_columns(cleaned, w, h, debug=debug)

    if len(points) < 5:
        return points

    # -------------------------------------------
    # Step 4: Savitzky-Golay smoothing
    # -------------------------------------------
    if len(points) > 15:
        y_raw = points[:, 1].astype(float)
        win = min(11, len(y_raw) - 1)
        if win % 2 == 0:
            win -= 1
        if win >= 5:
            y_smooth = savgol_filter(y_raw, window_length=win, polyorder=2)
            points[:, 1] = y_smooth

    # -------------------------------------------
    # Step 5: Skip calibration pulse at start
    # (narrow tall rectangle right after the label)
    # -------------------------------------------
    if len(points) > 30:
        head = points[:15, 1]
        if head.max() - head.min() > h * 0.6:
            points = points[15:]

    if debug:
        print(f"  Traced {len(points)} columns, "
              f"span {points[:,1].max() - points[:,1].min():.0f} px")

    return points


def _trace_columns(mask, w, h, debug=False):
    """
    Trace per column, picking the CLUSTER of dark pixels nearest to prev_y.
    """
    points = []
    prev_y = None
    continuity = max(30, h // 3)

    for col in range(w):
        rows = np.where(mask[:, col] > 0)[0]
        if len(rows) == 0:
            if prev_y is not None:
                points.append([col, prev_y])
            continue

        # Find runs of consecutive rows (clusters)
        runs = []
        start = rows[0]
        prev = rows[0]
        for r in rows[1:]:
            if r - prev <= 2:
                prev = r
            else:
                runs.append((start, prev))
                start = r
                prev = r
        runs.append((start, prev))

        # Pick the run nearest to prev_y
        if prev_y is None:
            best = max(runs, key=lambda r: r[1] - r[0])
            y = (best[0] + best[1]) // 2
        else:
            best_run = None
            best_dist = 1e9
            for r0, r1 in runs:
                run_center = (r0 + r1) // 2
                dist = abs(run_center - prev_y)
                if dist < best_dist:
                    best_dist = dist
                    best_run = (r0, r1)
            if best_dist > continuity:
                y = prev_y
            else:
                y = (best_run[0] + best_run[1]) // 2

        points.append([col, y])
        prev_y = y

    if debug:
        print(f"  Traced {len(points)} / {w} columns")

    return np.array(points) if points else np.empty((0, 2))


def trace_all_leads(image, lead_regions):
    """Trace multiple leads."""
    result = {}
    for lead_name, region in lead_regions.items():
        points = trace_waveform(image, region)
        result[lead_name] = points
    return result


if __name__ == "__main__":
    import sys
    from src.ecg.image_loader import load_ecg_image
    from src.ecg.image_to_signal import _detect_lead_regions

    img = load_ecg_image(sys.argv[1] if len(sys.argv) > 1 else "sample2.jpg")
    regions = _detect_lead_regions(img, "auto")

    print("Debug trace per lead:")
    print("=" * 70)
    for name, region in regions.items():
        print(f"\n{name}:")
        pts = trace_waveform(img, region, debug=True)
        if len(pts) > 0:
            y_min = pts[:, 1].min()
            y_max = pts[:, 1].max()
            print(f"  y-range: {y_min:.0f} to {y_max:.0f} (span {y_max - y_min:.0f} px)")