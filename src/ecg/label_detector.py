"""Optional OCR-based ECG lead-label localization."""

import csv
import io
import logging
import re
import shutil
import subprocess
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_LEADS = {"I", "II", "III", "AVR", "AVL", "AVF"}
_CHEST_LEADS = {f"V{i}" for i in range(1, 7)}


def detect_lead_labels(image: np.ndarray) -> Dict[str, dict]:
    """Return OCR label boxes when enough labels form a coherent ECG layout."""
    if shutil.which("tesseract") is None:
        logger.info("Tesseract is unavailable; skipping OCR")
        return {}

    variants = [_prepare_image(image, mode) for mode in ("gray", "threshold")]
    detections: List[dict] = []
    for variant in variants:
        detections.extend(_run_tesseract(variant))

    labels = _deduplicate_labels(detections)
    if len(labels) < 8:
        logger.info("OCR found only %d usable lead labels", len(labels))
        return {}

    regions = _build_regions(labels, image.shape[:2])
    if len(regions) < 8:
        logger.info("OCR labels did not form a coherent ECG layout")
        return {}
    return regions


def _prepare_image(image: np.ndarray, mode: str) -> np.ndarray:
    rgb = image.copy()
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = (
        cv2.inRange(hsv, np.array([0, 25, 80]), np.array([12, 255, 255]))
        | cv2.inRange(hsv, np.array([168, 25, 80]), np.array([180, 255, 255]))
    )
    rgb[red > 0] = 255
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    if mode == "threshold":
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 11,
        )
    return gray


def _run_tesseract(image: np.ndarray) -> List[dict]:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return []
    command = [
        "tesseract", "stdin", "stdout", "--psm", "11", "tsv",
        "-c", "tessedit_char_whitelist=IAV123Fvarl",
    ]
    try:
        result = subprocess.run(
            command,
            input=encoded.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    detections = []
    text_stream = io.StringIO(result.stdout.decode("utf-8", errors="ignore"))
    for row in csv.DictReader(text_stream, delimiter="\t"):
        text = _normalize_label(row.get("text", ""))
        try:
            confidence = float(row.get("conf", "-1"))
            left = int(row["left"]) // 2
            top = int(row["top"]) // 2
            width = int(row["width"]) // 2
            height = int(row["height"]) // 2
        except (TypeError, ValueError, KeyError):
            continue
        if text and confidence >= 10 and width > 2 and height > 2:
            detections.append({
                "label": text,
                "x": left,
                "y": top,
                "width": width,
                "height": height,
                "confidence": confidence,
            })
    return detections


def _normalize_label(value: str) -> Optional[str]:
    text = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    aliases = {
        "1": "I", "L": "I", "IL": "II", "ILL": "III",
        "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
        "VI": "V1", "V2": "V2", "V3": "V3", "V4": "V4",
        "V5": "V5", "V6": "V6",
    }
    if text in aliases:
        return aliases[text]
    if text in _LEADS:
        return {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}.get(text, text)
    if text in _CHEST_LEADS:
        return text
    return None


def _deduplicate_labels(detections: List[dict]) -> List[dict]:
    selected = {}
    for item in sorted(detections, key=lambda value: value["confidence"], reverse=True):
        label = item["label"]
        center = (item["x"], item["y"])
        if any(
            abs(center[0] - other["x"]) < 25 and abs(center[1] - other["y"]) < 20
            for other in selected.values()
        ):
            continue
        selected[f"{label}:{center}"] = item
    return list(selected.values())


def _build_regions(labels: List[dict], image_shape) -> Dict[str, dict]:
    height, width = image_shape
    rows = []
    for label in sorted(labels, key=lambda item: item["y"]):
        center_y = label["y"] + label["height"] / 2
        row = next((current for current in rows if abs(current[0] - center_y) < height * 0.035), None)
        if row is None:
            row = [center_y, []]
            rows.append(row)
        row[1].append(label)

    rows = [sorted(row[1], key=lambda item: item["x"]) for row in rows]
    rows = [row for row in rows if len(row) >= 2]
    if len(rows) < 5 or sum(len(row) >= 2 for row in rows) < 5:
        return {}

    expected = [
        ["I", "V1"], ["II", "V2"], ["III", "V3"],
        ["aVR", "V4"], ["aVL", "V5"], ["aVF", "V6"],
    ]
    regions = {}
    for row_index, row in enumerate(rows[:6]):
        if row_index >= len(expected):
            break
        row_top = max(0, min(item["y"] for item in row) - 20)
        row_bottom = (
            min(item["y"] for item in rows[row_index + 1]) - 10
            if row_index + 1 < len(rows)
            else int(height * 0.95)
        )
        if row_bottom <= row_top:
            continue
        for column, item in enumerate(row[:2]):
            label_name = expected[row_index][column]
            x_start = item["x"] + item["width"] + 8
            x_end = row[column + 1]["x"] - 8 if column + 1 < len(row) else int(width * 0.98)
            if x_end - x_start >= 100:
                regions[label_name] = {
                    "x": x_start,
                    "y": row_top,
                    "width": x_end - x_start,
                    "height": row_bottom - row_top,
                }
    return regions
