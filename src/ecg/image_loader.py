"""
Load ECG images from various formats.
Enhances low-quality / photographed images and normalizes to 2000x1400.
"""

from pathlib import Path
from typing import Union
import numpy as np
import cv2
from PIL import Image

try:
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


TARGET_WIDTH = 2000
TARGET_HEIGHT = 1400


def load_ecg_image(path, normalize=True):
    """Load ECG image, enhance if low quality, resize to standard size."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Image not found: {}".format(path))

    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
        img = _load_raster(path)
    elif suffix == ".pdf":
        if not PDF_SUPPORT:
            raise ImportError("PDF support requires: pip install PyMuPDF")
        img = _load_pdf(path)
    else:
        raise ValueError("Unsupported format: {}".format(suffix))

    if normalize:
        img = _normalize_image(img)

    return img


def _load_raster(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)


def _load_pdf(path):
    doc = fitz.open(str(path))
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)
    doc.close()
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def _normalize_image(img):
    """
    Enhance + resize image to standard size.

    Enhancement:
    1. Upscale small images (min dimension < 1000) to make traces clearer
    2. Apply CLAHE (contrast limited adaptive histogram equalization)
       for better trace visibility in photographs
    3. Resize to TARGET_WIDTH x TARGET_HEIGHT with padding
    """
    h, w = img.shape[:2]

    # 1. Upscale if too small
    min_dim = min(h, w)
    if min_dim < 1000:
        scale_up = 1000.0 / min_dim
        new_w = int(w * scale_up)
        new_h = int(h * scale_up)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]

    # 2. Contrast enhancement (CLAHE on LAB L-channel)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    # 3. Resize to target size (letterbox)
    target_w = TARGET_WIDTH
    target_h = TARGET_HEIGHT
    scale = min(target_w / float(w), target_h / float(h))
    new_w = int(w * scale)
    new_h = int(h * scale)

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # 4. Pad with white
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    return padded


def get_image_info(path):
    img = load_ecg_image(path)
    return {
        "path": str(path),
        "shape": img.shape,
        "height": img.shape[0],
        "width": img.shape[1],
        "channels": img.shape[2],
        "dtype": str(img.dtype),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        info = get_image_info(sys.argv[1])
        for k, v in info.items():
            print("{}: {}".format(k, v))
    else:
        print("Usage: python -m src.ecg.image_loader <image_path>")