"""Quality checks for ECG images before waveform extraction."""

import cv2
import numpy as np


def assess_image_quality(image: np.ndarray, min_score: float = 0.20) -> dict:
    """Return content-based quality measurements and an extraction decision."""
    issues = []
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        return {
            "valid": False,
            "passed": False,
            "score": 0.0,
            "issues": ["expected an RGB image with shape (H, W, 3)"],
        }
    if image.shape[0] < 100 or image.shape[1] < 100:
        issues.append("image is too small")
    if not np.isfinite(image).all():
        issues.append("image contains non-finite values")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    contrast = float(gray.std() / 64.0)
    contrast_score = float(np.clip(contrast, 0.0, 1.0))
    non_white = float(np.mean(gray < 245))
    dark_fraction = float(np.mean(gray < 130))
    red_fraction = float(
        np.mean(
            (cv2.inRange(hsv, np.array([0, 30, 100]), np.array([10, 255, 255])) > 0)
            | (cv2.inRange(hsv, np.array([170, 30, 100]), np.array([180, 255, 255])) > 0)
        )
    )
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    if contrast_score < 0.08:
        issues.append("very low contrast")
    if non_white < 0.005:
        issues.append("image contains almost no content")
    if dark_fraction < 0.0005:
        issues.append("no substantial dark waveform or text detected")
    if edge_density < 0.001:
        issues.append("very few image edges detected")

    content_score = float(np.clip(non_white / 0.20, 0.0, 1.0))
    ink_score = float(np.clip(dark_fraction / 0.02, 0.0, 1.0))
    edge_score = float(np.clip(edge_density / 0.05, 0.0, 1.0))
    score = float(
        np.clip(0.45 * contrast_score + 0.25 * content_score + 0.20 * ink_score + 0.10 * edge_score, 0.0, 1.0)
    )
    valid = len(issues) == 0 or not any(
        issue in issues
        for issue in ("image is too small", "image contains non-finite values", "image contains almost no content")
    )
    return {
        "valid": valid,
        "passed": bool(valid and score >= min_score),
        "score": score,
        "contrast": float(gray.std()),
        "non_white_fraction": non_white,
        "dark_fraction": dark_fraction,
        "red_grid_fraction": red_fraction,
        "edge_density": edge_density,
        "issues": issues,
    }
