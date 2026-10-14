"""
Visualize the trace mask for each detected lead region.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.ecg.image_loader import load_ecg_image
from src.ecg.image_to_signal import _detect_lead_regions


IMAGE_PATH = "sample2.jpg"

img = load_ecg_image(IMAGE_PATH)
regions = _detect_lead_regions(img, "12-lead-standard")

print(f"Detected {len(regions)} regions")

fig, axes = plt.subplots(4, 3, figsize=(18, 14))
axes = axes.flatten()

for i, (name, r) in enumerate(list(regions.items())[:12]):
    x, y, w, h = r["x"], r["y"], r["width"], r["height"]
    crop = img[y:y+h, x:x+w]

    # Same mask as the tracer uses (V<120, S<85)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    mask = ((V < 120) & (S < 85)).astype(np.uint8) * 255

    axes[i].imshow(mask, cmap='gray', aspect='auto')
    axes[i].set_title(f"{name} — mask ({mask.sum()//255} px) region w={w}h={h}")
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("debug_masks.png", dpi=90)
plt.show()
print("Saved debug_masks.png")