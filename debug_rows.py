"""
Visualize the row projection to see why bands aren't separating.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening

from src.ecg.image_loader import load_ecg_image

IMAGE = "sample2.jpg"

img = load_ecg_image(IMAGE)
h, w = img.shape[:2]
print(f"Image: {img.shape}")

# Dark ink mask
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
S = hsv[:, :, 1]
V = hsv[:, :, 2]
ink = ((V < 130) & (S < 90)).astype(np.uint8)

# Long horizontal ink (for header detection)
long_horiz = binary_opening(
    ink, structure=np.ones((1, 40), dtype=bool)
).astype(np.uint8)

row_long = long_horiz.sum(axis=1)
row_ink = ink.sum(axis=1).astype(float)

# Smooth
k = np.ones(11) / 11
row_smooth = np.convolve(row_ink, k, mode='same')

# Print stats
print(f"\nMax row ink (raw): {row_ink.max()}")
print(f"Min row ink (raw): {row_ink.min()}")
print(f"Max row ink (smooth): {row_smooth.max():.0f}")
print(f"30% of max (smooth): {row_smooth.max() * 0.3:.0f}")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(row_ink, label='raw ink per row', color='steelblue')
axes[0].plot(row_smooth, label='smoothed', color='red')
axes[0].axhline(row_smooth.max() * 0.3, color='green', linestyle='--',
                label=f'30% threshold ({row_smooth.max()*0.3:.0f})')
axes[0].set_ylabel("Dark pixels")
axes[0].set_title(f"Row projection — {IMAGE}")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].invert_yaxis()  # y increases downward

# Show the mask as an image
axes[1].imshow(ink, cmap='gray', aspect='auto')
axes[1].set_ylabel("Row index")
axes[1].set_xlabel("Column index")
axes[1].set_title("Dark ink mask")

plt.tight_layout()
plt.savefig("debug_rows.png", dpi=90)
plt.show()
print("\nSaved debug_rows.png")