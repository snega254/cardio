"""
Full diagnostic: show mask, row projection, and column projections.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening
from src.ecg.image_loader import load_ecg_image

img = load_ecg_image("sample2.jpg")
h, w = img.shape[:2]

hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
S = hsv[:, :, 1]
V = hsv[:, :, 2]
ink = ((V < 130) & (S < 90)).astype(np.uint8)

# Remove borders
border_y = 40
border_x = 40
ink[:border_y, :] = 0
ink[-border_y:, :] = 0
ink[:, :border_x] = 0
ink[:, -border_x:] = 0

# Row projection
row_proj = ink.sum(axis=1)

fig = plt.figure(figsize=(18, 14))

# Panel 1: ink mask
ax1 = plt.subplot(3, 1, 1)
ax1.imshow(ink, cmap='gray', aspect='auto')
ax1.set_title('Ink mask (borders removed)')
ax1.set_ylabel('row index')

# Panel 2: row projection (raw)
ax2 = plt.subplot(3, 1, 2)
ax2.plot(row_proj, np.arange(h), label='raw row ink')
ax2.axhline(0, color='k', alpha=0.3)
ax2.invert_yaxis()
ax2.set_title(f'Row projection: max={row_proj.max()}, min={row_proj.min()}, median={int(np.median(row_proj))}')
ax2.set_ylabel('row index')
ax2.set_xlabel('dark pixels in this row')
ax2.grid(alpha=0.3)
ax2.legend()

# Mark the 6 equal row centers
usable = h - 2*border_y
for r in range(6):
    y_center = border_y + int(usable * (r + 0.5) / 6)
    ax2.axhline(y_center, color='red', alpha=0.5, linestyle='--')

# Panel 3: column projections for 6 equal rows
for r in range(6):
    y0 = border_y + r * usable // 6
    y1 = border_y + (r + 1) * usable // 6
    row_ink = ink[y0:y1, :]
    col_proj = row_ink.sum(axis=0)
    ax3 = plt.subplot(6, 1, r + 1) if r < 6 else None
    if r < 3:
        pass

# Instead, print stats
print("\n=== Row ink stats ===")
print(f"max: {row_proj.max()}")
print(f"min: {row_proj.min()}")
print(f"median: {int(np.median(row_proj))}")

print("\n=== Ink per equal row band (6 bands) ===")
for r in range(6):
    y0 = border_y + r * usable // 6
    y1 = border_y + (r + 1) * usable // 6
    band_ink = ink[y0:y1, :].sum()
    print(f"  Band {r} (y={y0}-{y1}): {band_ink} dark pixels")

print("\n=== Ink per equal column band (2 columns) ===")
for c in range(2):
    x0 = c * w // 2
    x1 = (c + 1) * w // 2
    col_ink = ink[:, x0:x1].sum()
    print(f"  Column {c} (x={x0}-{x1}): {col_ink} dark pixels")

plt.tight_layout()
plt.savefig("debug_full.png", dpi=90)
plt.show()
print("\nSaved debug_full.png")