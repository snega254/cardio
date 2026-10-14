
import numpy as np
from src.ecg.interface import predict
from src.ecg.image_to_signal import extract_signal_from_image

# Replace this with the path to an ECG image you have already tested
ECG_IMAGE_PATH = r"D:\ccardio\cardio\sample.png"

print("Extracting ECG signal...")

result = extract_signal_from_image(ECG_IMAGE_PATH)

if not result.get("success"):
    print("ECG extraction failed:", result)
    raise SystemExit(1)

signal = result["signal"]
print("Signal shape:", signal.shape)

print("\nRunning prediction with Grad-CAM...")

predicted_class, confidence, cam = predict(signal)

print("Predicted class:", predicted_class)
print("Confidence:", round(confidence, 4))

if cam is None:
    print("FAIL: Grad-CAM map is None")
else:
    cam = np.asarray(cam)

    print("Grad-CAM shape:", cam.shape)
    print("All values finite:", np.isfinite(cam).all())
    print("Minimum:", float(cam.min()))
    print("Maximum:", float(cam.max()))
    print("All zeros:", bool(np.all(cam == 0)))

    if (
        cam.ndim == 1
        and len(cam) == signal.shape[0]
        and np.isfinite(cam).all()
        and cam.max() > cam.min()
    ):
        print("\nPASS: Grad-CAM map was generated.")
    else:
        print("\nCHECK: Grad-CAM output needs further investigation.")
