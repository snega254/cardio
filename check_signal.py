"""
Visualize the image-to-signal pipeline for a given ECG image.
"""

import sys
from pathlib import Path
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt

from src.ecg.image_to_signal import extract_signal_from_image
from src.ecg.preprocessing import preprocess_signal
from src.ecg.dataset import IDX_TO_LABEL


IMAGE_PATH = "sample2.jpg"   # <-- change this to your test image


print(f"Loading: {IMAGE_PATH}")
result = extract_signal_from_image(IMAGE_PATH)

if not result.get("success"):
    print("Extraction failed:", result.get("message"))
    sys.exit(1)

raw_signal = result["signal"]
print(f"\nRaw signal shape: {raw_signal.shape}")
print(f"Raw dtype: {raw_signal.dtype}")
print(f"Raw min/max: {raw_signal.min():.4f} / {raw_signal.max():.4f}")
print(f"Raw NaN count: {np.isnan(raw_signal).sum()}")
print(f"Raw zeros count: {(raw_signal == 0).sum()} of {raw_signal.size}")

# Per-lead stats
print("\nPer-lead stats (raw):")
for i in range(raw_signal.shape[1]):
    lead = raw_signal[:, i]
    print(f"  Lead {i}: min={lead.min():+.4f}  max={lead.max():+.4f}  "
          f"mean={lead.mean():+.4f}  std={lead.std():.4f}  nan={np.isnan(lead).sum()}")


# Preprocess
processed = preprocess_signal(raw_signal)
print(f"\nProcessed shape: {processed.shape}")
print(f"Processed min/max: {processed.min():.4f} / {processed.max():.4f}")
print(f"Processed NaN count: {np.isnan(processed).sum()}")


# Plot raw + processed side by side
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
fig.suptitle(f"Image-to-Signal: {IMAGE_PATH}", fontsize=14)

fs = 100  # PTB-XL uses 100 Hz
t = np.arange(raw_signal.shape[0]) / fs

for i in range(4):
    # Raw
    axes[0, i].plot(t, raw_signal[:, i], linewidth=0.8, color='steelblue')
    axes[0, i].set_title(f"Raw Lead {i}")
    axes[0, i].grid(True, alpha=0.3)
    axes[0, i].set_ylabel("mV")

    # Processed
    axes[1, i].plot(t, processed[:, i], linewidth=0.8, color='darkred')
    axes[1, i].set_title(f"Processed Lead {i}")
    axes[1, i].grid(True, alpha=0.3)
    axes[1, i].set_ylabel("normalized")
    axes[1, i].set_xlabel("time (s)")

plt.tight_layout()
plt.savefig("signal_check.png", dpi=100)
plt.show()

print("\nSaved plot to: signal_check.png")