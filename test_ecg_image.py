"""Run the ECG image-to-signal pipeline on one image."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image

from src.ecg.image_to_signal import extract_signal_from_image


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resize an ECG image and run ECG-Digitiser on it."
    )
    parser.add_argument(
        "image",
        help="Path to the ECG image, for example sample2.jpg",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1 / 3,
        help="Resize factor before digitization (default: 1/3).",
    )
    parser.add_argument(
        "--output",
        help="Path for the resized image. Defaults to <stem>_small.jpg.",
    )
    args = parser.parse_args()

    source = Path(args.image).expanduser().resolve()
    if not source.is_file():
        print(f"Image not found: {source}")
        return 1
    if not 0 < args.scale <= 1:
        print("--scale must be greater than 0 and no greater than 1.")
        return 1

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else source.with_name(f"{source.stem}_small.jpg")
    )

    with Image.open(source) as image:
        original_size = image.size
        width = max(1, round(image.width * args.scale))
        height = max(1, round(image.height * args.scale))
        resized = image.convert("RGB").resize((width, height))
        resized.save(output, format="JPEG", quality=95)

    print(f"Original: {source.name} ({original_size[0]} x {original_size[1]})")
    print(f"Resized:  {output.name} ({width} x {height})")
    print("Running ECG-Digitiser...")

    started = time.time()
    result = extract_signal_from_image(output)
    elapsed = time.time() - started

    print(f"Elapsed: {elapsed:.2f} seconds")
    print(f"Success: {result.get('success')}")
    print(f"Error:   {result.get('error', '')}")

    signal = result.get("signal")
    if signal is not None:
        print(f"Signal shape: {signal.shape}")
        print(f"Leads: {result.get('leads')}")
        print(f"Sampling rate: {result.get('sampling_rate')} Hz")
        print(f"Duration: {result.get('duration_seconds')} seconds")

    metadata = result.get("metadata") or {}
    if metadata.get("digitiser_return_code") not in (None, 0):
        print(f"Return code: {metadata['digitiser_return_code']}")
        print("\nDigitiser stdout:")
        print(metadata.get("digitiser_stdout", ""))
        print("\nDigitiser stderr:")
        print(metadata.get("digitiser_stderr", ""))

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
