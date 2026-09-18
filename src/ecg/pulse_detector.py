"""
Calibration-pulse detector for ECG lead localization.

Every printed ECG has a square 10mm x 2mm calibration pulse at the start
of each lead trace. Detecting these pulses gives the exact x-position
where each lead's waveform begins, without relying on OCR.
"""

import logging
from typing import List, Dict, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _trace_mask(image: np.ndarray) -> np.ndarray:
    """Return a binary mask of dark, achromatic (ink) pixels only."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    mask = ((V < 130) & (S < 90)).astype(np.uint8) * 255
    return mask


def detect_pulses(
    image: np.ndarray,
    grid_px_per_mm: float,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect calibration pulses with strict size + shape filtering.

    Returns list of (x, y, w, h) bounding boxes, sorted by (row_y, x).
    """
    h, w = image.shape[:2]
    mask = _trace_mask(image)

    px_per_mm = max(1.0, float(grid_px_per_mm))

    # --- Tight size window (physical units) ---
    # Calibration pulse is 10 mm tall x 2 mm wide.
    target_h = px_per_mm * 10
    target_w = px_per_mm * 2.0
    min_h = int(target_h * 0.6)      # 6 mm
    max_h = int(target_h * 1.4)      # 14 mm
    min_w = int(target_w * 0.5)      # 1 mm
    max_w = int(target_w * 2.0)      # 4 mm

    logger.info(
        f"Pulse size window: h={min_h}-{max_h}px  w={min_w}-{max_w}px"
    )

    # --- Remove long horizontal trace segments ---
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horiz_kernel)
    mask_no_trace = cv2.subtract(mask, horiz)

    # --- Find connected components ---
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_no_trace, connectivity=8
    )

    candidates = []
    for i in range(1, num):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        # Size filter
        if not (min_h <= hh <= max_h):
            continue
        if not (min_w <= ww <= max_w):
            continue

        # Printed pulses are often broken outlines after resizing and grid
        # removal, so fill ratio is only a weak rejection signal.
        fill = area / float(ww * hh + 1e-6)
        if fill < 0.08:
            continue

        # Tolerate perspective, broken edges, and anti-aliasing.
        ar = hh / float(ww + 1e-6)
        if ar < 2.5 or ar > 14.0:
            continue

        candidates.append((x, y, ww, hh))

    logger.info(f"Pulse-like candidates: {len(candidates)}")

    if not candidates:
        return []

    # --- Group into rows by y ---
    candidates.sort(key=lambda b: b[1])
    row_tol = int(px_per_mm * 30)
    grouped_rows = []
    current_row = [candidates[0]]
    for p in candidates[1:]:
        if abs(p[1] - current_row[0][1]) < row_tol:
            current_row.append(p)
        else:
            grouped_rows.append(current_row)
            current_row = [p]
    if current_row:
        grouped_rows.append(current_row)

    logger.info(f"Grouped rows: {len(grouped_rows)}")

    # Keep rows with the expected structure when available. A single
    # candidate can still be useful for a one-column layout.
    valid_rows = [r for r in grouped_rows if 1 <= len(r) <= 6]

    logger.info(f"Valid rows (2-6 pulses): {len(valid_rows)}")

    final_pulses = [p for row in valid_rows for p in row]
    final_pulses.sort(key=lambda b: (b[1], b[0]))

    return final_pulses


def group_pulses_into_rows(
    pulses: List[Tuple[int, int, int, int]],
    image_height: int,
) -> List[List[Tuple[int, int, int, int]]]:
    """Group pulses into rows by y-coordinate."""
    if not pulses:
        return []

    row_tol = image_height * 0.06
    rows = []
    current_row = [pulses[0]]

    for p in pulses[1:]:
        if abs(p[1] - current_row[0][1]) < row_tol:
            current_row.append(p)
        else:
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [p]

    if current_row:
        rows.append(sorted(current_row, key=lambda b: b[0]))

    return rows


def build_regions_from_pulses(
    pulses_by_row: List[List[Tuple[int, int, int, int]]],
    image_shape: Tuple[int, int],
) -> Dict[str, dict]:
    """
    Convert pulse positions into lead regions using standard ECG reading order.

    Reading order (universal across ECG printers):
        I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    Row-major, top-to-bottom, left-to-right.
    """
    h, w = image_shape

    if len(pulses_by_row) >= 5 and max(len(row) for row in pulses_by_row) >= 2:
        row_major_leads = [
            ["I", "V1"], ["II", "V2"], ["III", "V3"],
            ["aVR", "V4"], ["aVL", "V5"], ["aVF", "V6"],
        ]
    else:
        row_major_leads = [[
            "I", "II", "III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6",
        ]]

    regions = {}
    idx = 0

    for r_idx, row in enumerate(pulses_by_row):
        if r_idx >= len(row_major_leads):
            break

        # Vertical extent of this row
        y_top = max(0, min(p[1] for p in row) - 20)
        if r_idx + 1 < len(pulses_by_row):
            y_bot = min(p[1] for p in pulses_by_row[r_idx + 1]) - 20
        else:
            y_bot = int(h * 0.95)

        for c_idx, pulse in enumerate(row):
            if r_idx >= len(row_major_leads) or c_idx >= len(row_major_leads[r_idx]):
                break

            px, py, pw, ph = pulse
            x_start = px + pw + 5

            if c_idx + 1 < len(row):
                x_end = row[c_idx + 1][0] - 5
            else:
                x_end = int(w * 0.98)

            if x_end - x_start < 50:
                continue

            regions[row_major_leads[r_idx][c_idx]] = {
                "x": x_start,
                "y": y_top,
                "width": x_end - x_start,
                "height": y_bot - y_top,
            }
            idx += 1

    logger.info(f"Built {len(regions)} lead regions from pulses")
    return regions


def detect_lead_regions_from_pulses(
    image: np.ndarray,
    grid_px_per_mm: float,
) -> Dict[str, dict]:
    """One-call helper: pulses -> rows -> regions."""
    pulses = detect_pulses(image, grid_px_per_mm)
    if len(pulses) < 4:
        return {}
    rows = group_pulses_into_rows(pulses, image.shape[0])

    # Reject isolated dark components that happen to resemble a pulse.
    # Tier 1 is only safe when the components form a recognizable layout.
    row_sizes = sorted((len(row) for row in rows), reverse=True)
    two_column_rows = sum(size >= 2 for size in row_sizes)
    four_column_rows = sum(size >= 4 for size in row_sizes)
    coherent_6x2 = len(rows) in (5, 6, 7) and two_column_rows >= 4
    coherent_3x4 = len(rows) in (3, 4) and four_column_rows >= 2
    if not (coherent_6x2 or coherent_3x4):
        logger.info("Pulse-like components do not form a coherent lead layout")
        return {}

    return build_regions_from_pulses(rows, image.shape[:2])


if __name__ == "__main__":
    import sys
    from src.ecg.image_loader import load_ecg_image
    from src.ecg.grid_detector import detect_grid

    path = sys.argv[1] if len(sys.argv) > 1 else "sample2.jpg"
    img = load_ecg_image(path)
    grid = detect_grid(img)

    print(f"Image: {img.shape}")
    print(f"Grid: {grid['small_box_px']:.2f} px/mm")

    pulses = detect_pulses(img, grid["small_box_px"])
    rows = group_pulses_into_rows(pulses, img.shape[0])

    print(f"\nPulses found: {len(pulses)}")
    print(f"Rows: {len(rows)}")
    for i, row in enumerate(rows):
        print(f"  Row {i}: {len(row)} pulses  y≈{row[0][1]}")

    regions = build_regions_from_pulses(rows, img.shape[:2])
    print(f"\nRegions built: {len(regions)}")
    for name, r in sorted(regions.items()):
        print(f"  {name:5s} x={r['x']:5d} y={r['y']:5d} w={r['width']:4d} h={r['height']:4d}")