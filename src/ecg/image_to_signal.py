"""
CardioAgent ECG image -> waveform extraction.

IMPORTANT:
- ECG-Digitiser is the ONLY ECG image digitization method used here.
- No custom waveform tracing.
- No OCR-based waveform extraction.
- No pulse/projection/grid based signal extraction.
- JPG/JPEG input is converted to PNG only because the official
  ECG-Digitiser expects IMAGE_TYPE="png".
- The digitizer's WFDB output is then read and prepared for the
  CardioAgent ECG classifier.
"""

from __future__ import annotations

import os
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.signal import resample_poly

import wfdb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CardioAgent model input configuration
# ---------------------------------------------------------------------------

TARGET_LEADS = [
    "I",
    "II",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

TARGET_FS = 100
TARGET_DURATION_SECONDS = 2.5
TARGET_LENGTH = 1000

# Official ECG-Digitiser configuration currently uses:
# FREQUENCY = 500
# IMAGE_TYPE = "png"
DIGITISER_FS = 500
DIGITISER_TIMEOUT_SECONDS = int(
    os.getenv("ECG_DIGITISER_TIMEOUT_SECONDS", "600")
)
DIGITISER_IMAGE_SCALE = float(
    os.getenv("ECG_DIGITISER_IMAGE_SCALE", "1.0")
)

DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "ecg_signal_cache"
)


# ---------------------------------------------------------------------------
# Optional feature extraction
# ---------------------------------------------------------------------------

try:
    from src.ecg.features import extract_ecg_features
except Exception:
    extract_ecg_features = None


# ---------------------------------------------------------------------------
# Locate ECG-Digitiser
# ---------------------------------------------------------------------------

def _find_external_digitiser_repo() -> Optional[Path]:
    """
    Locate the cloned official ECG-Digitiser repository.

    Search order:
    1. ECG_DIGITISER_PATH environment variable
    2. <project root>/ECG-Digitiser
    3. current working directory/ECG-Digitiser
    """

    env_path = os.getenv("ECG_DIGITISER_PATH")

    if env_path:
        candidate = Path(env_path).expanduser().resolve()

        if (candidate / "src" / "run" / "digitize.py").exists():
            return candidate

    project_root = Path(__file__).resolve().parents[2]

    candidate = project_root / "ECG-Digitiser"

    if (candidate / "src" / "run" / "digitize.py").exists():
        return candidate

    candidate = Path.cwd() / "ECG-Digitiser"

    if (candidate / "src" / "run" / "digitize.py").exists():
        return candidate

    return None


def _find_digitiser_model(repo_root: Path) -> Optional[Path]:
    """
    Locate the ECG-Digitiser M3 model directory.
    """

    env_model = os.getenv("ECG_DIGITISER_MODEL")

    if env_model:
        candidate = Path(env_model).expanduser().resolve()

        if (
            candidate / "nnUNet_results"
        ).exists():
            return candidate

    candidate = repo_root / "models" / "M3"

    if (candidate / "nnUNet_results").exists():
        return candidate

    return None


# ---------------------------------------------------------------------------
# Input image preparation
# ---------------------------------------------------------------------------

def _prepare_png_input(
    image_path: str | Path,
    input_dir: Path,
) -> Path:
    """
    Convert the supplied ECG image to PNG.

    ECG-Digitiser/config.py specifies:

        IMAGE_TYPE = "png"

    Therefore the temporary input supplied to ECG-Digitiser must have
    a .png extension.

    This function performs ONLY image-format conversion. It does not
    perform any ECG waveform extraction or signal processing.
    """

    source = Path(image_path).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(
            f"ECG image not found: {source}"
        )

    if not source.is_file():
        raise ValueError(
            f"ECG image path is not a file: {source}"
        )

    input_dir.mkdir(parents=True, exist_ok=True)

    output_path = input_dir / "ecg_input.png"

    try:
        with Image.open(source) as image:
            # ECG-Digitiser ultimately reads the image using torchvision.
            # RGB keeps the image compatible with its expected image input.
            image = image.convert("RGB")
            scale = DIGITISER_IMAGE_SCALE
            if not 0 < scale <= 1:
                raise ValueError(
                    "ECG_DIGITISER_IMAGE_SCALE must be greater than 0 "
                    "and no greater than 1."
                )

            # IMPORTANT: the pretrained ECG-Digitiser segmentation model
            # expects the ECG page at approximately the original resolution.
            # Downscaling the page to 1/3 can erase thin ECG traces and lead
            # labels, causing the digitizer to return incomplete leads.
            # Keep the original resolution by default (scale=1.0).
            if scale != 1.0:
                width = max(1, round(image.width * scale))
                height = max(1, round(image.height * scale))
                image = image.resize(
                    (width, height),
                    Image.Resampling.LANCZOS,
                )

            image.save(output_path, format="PNG")

    except Exception as exc:
        raise RuntimeError(
            f"Could not convert ECG image to PNG: {source}"
        ) from exc

    if not output_path.exists():
        raise RuntimeError(
            "PNG conversion completed without creating the input file."
        )

    return output_path


def _cache_path(image_path: Path) -> Path:
    """Return a stable cache path based on the image contents and model input."""
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    cache_dir = Path(
        os.getenv("ECG_SIGNAL_CACHE_DIR", str(DEFAULT_CACHE_DIR))
    ).expanduser()
    return cache_dir / f"{digest}_{TARGET_FS}hz_{TARGET_LENGTH}x{len(TARGET_LEADS)}.npz"


def _load_cached_signal(image_path: Path) -> Optional[Dict[str, Any]]:
    """Load a previously digitized signal, if a valid cache entry exists."""
    cache_path = _cache_path(image_path)
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            signal = np.asarray(cached["signal"], dtype=np.float32)
            features_json = str(cached["features"].item())
            features = json.loads(features_json)

        expected_shape = (TARGET_LENGTH, len(TARGET_LEADS))
        if signal.shape != expected_shape or not np.isfinite(signal).all():
            raise ValueError(f"cached signal has invalid shape or values: {signal.shape}")

        logger.info("Using cached ECG signal: %s", cache_path)
        return {
            "success": True,
            "signal": signal,
            "leads": list(TARGET_LEADS),
            "sampling_rate": TARGET_FS,
            "duration_seconds": TARGET_DURATION_SECONDS,
            "features": features,
            "metadata": {
                "cache_hit": True,
                "cache_path": str(cache_path),
            },
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring invalid ECG signal cache %s: %s", cache_path, exc)
        return None


def _save_cached_signal(
    image_path: Path,
    signal: np.ndarray,
    features: Dict[str, Any],
) -> None:
    """Persist a successful digitized signal for reuse on later requests."""
    cache_path = _cache_path(image_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_path,
        signal=np.asarray(signal, dtype=np.float32),
        features=np.array(json.dumps(features, default=str)),
    )
    os.replace(temporary_path, cache_path)


# ---------------------------------------------------------------------------
# Signal validation  ── FIX 1 ──
# ---------------------------------------------------------------------------

def _validate_raw_signal(
    raw_signal: np.ndarray,
    signal_names: list,
    source_fs: float,
) -> str:
    """
    Return '' if the raw digitiser signal looks usable.
    Otherwise return a human-readable reason it must be rejected.

    Rejects:
      - signals far shorter than the expected duration (time axis unreliable)
      - leads that are mostly exactly-zero (NaN-padded by the digitiser)
      - leads that are flat (std ~ 0)
      - leads with huge discontinuities (NaN->0 boundary artifacts)
    """
    if raw_signal.ndim != 2 or raw_signal.shape[1] == 0:
        return f"empty signal shape {raw_signal.shape}"

    # 1) Duration / sample-count check
    expected = int(source_fs * TARGET_DURATION_SECONDS)
    if expected > 0 and raw_signal.shape[0] < 0.7 * expected:
        return (
            f"only {raw_signal.shape[0]} samples at {source_fs} Hz "
            f"(expected ~{expected}) — time axis unreliable"
        )

    # 2) Per-lead zero-fraction + flatness
    for i, name in enumerate(signal_names):
        col = raw_signal[:, i].astype(np.float64)
        zero_frac = float(np.mean(np.isclose(col, 0.0, atol=1e-6)))
        if zero_frac > 0.05:
            return (
                f"lead {name} is {zero_frac:.0%} zeros — "
                f"NaN padding from digitiser"
            )
        if float(np.nanstd(col)) < 1e-4:
            return f"lead {name} is flat (std={float(np.nanstd(col)):.2e})"

    # 3) Discontinuity check
        # 3) Discontinuity check
    #    QRS spikes are legitimately 30-100x the baseline noise, so a
    #    50x threshold produces false positives. A genuine digitiser
    #    glitch (trace jumping across the page) is typically >500x.
    #    We use 200x as a compromise.
    for i, name in enumerate(signal_names):
        col = raw_signal[:, i].astype(np.float64)
        d = np.abs(np.diff(col))
        nonzero = d[d > 0]
        if nonzero.size == 0:
            continue
        med = float(np.median(nonzero))
        if med > 0 and float(d.max()) > 200 * med:
            return (
                f"lead {name} has a discontinuity "
                f"(max step {float(d.max()):.3f}, median {med:.3f})"
            )

    return ""


# ---------------------------------------------------------------------------
# Run official ECG-Digitiser
# ---------------------------------------------------------------------------

def _run_external_digitiser(
    image_path: str | Path,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """
    Run the official ECG-Digitiser pipeline.

    Returns:
        (WFDB header path, metadata)

    No custom digitization is performed here.
    """

    repo_root = _find_external_digitiser_repo()

    if repo_root is None:
        return None, {
            "success": False,
            "error": (
                "ECG-Digitiser repository was not found. "
                "Expected ECG-Digitiser/src/run/digitize.py."
            ),
        }

    digitize_script = repo_root / "src" / "run" / "digitize.py"

    if not digitize_script.exists():
        return None, {
            "success": False,
            "error": (
                f"ECG-Digitiser entry point not found: "
                f"{digitize_script}"
            ),
        }

    model_dir = _find_digitiser_model(repo_root)

    if model_dir is None:
        return None, {
            "success": False,
            "error": (
                "ECG-Digitiser M3 model was not found. "
                "Expected ECG-Digitiser/models/M3/nnUNet_results."
            ),
        }

    temp_root = Path(
        tempfile.mkdtemp(prefix="cardioagent_digitiser_")
    )

    input_dir = temp_root / "input"
    output_dir = temp_root / "output"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        png_input = _prepare_png_input(
            image_path,
            input_dir,
        )

        png_files = list(input_dir.glob("*.png"))

        if not png_files:
            return None, {
                "success": False,
                "error": (
                    "PNG input was not created. "
                    "ECG-Digitiser requires .png input."
                ),
            }

        command = [
            sys.executable,
            "-m",
            "src.run.digitize",
            "-d",
            str(input_dir),
            "-m",
            str(model_dir),
            "-o",
            str(output_dir),
            "-v",
        ]

        env = os.environ.copy()

        env["PYTHONPATH"] = os.pathsep.join(
            [
                str(repo_root),
                env.get("PYTHONPATH", ""),
            ]
        ).rstrip(os.pathsep)

        env["PATH"] = os.pathsep.join(
            [
                str(Path(sys.executable).parent),
                env.get("PATH", ""),
            ]
        )

        env["nnUNet_results"] = str(model_dir / "nnUNet_results")

        try:
            completed = subprocess.run(
                command,
                cwd=str(repo_root),
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=DIGITISER_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            metadata = {
                "success": False,
                "error": (
                    f"ECG-Digitiser timed out after {DIGITISER_TIMEOUT_SECONDS} seconds. "
                    "This usually means the model is large, the input is invalid, or the external digitizer is hanging."
                ),
                "digitiser_command": " ".join(
                    f'"{str(part)}"' if " " in str(part) else str(part)
                    for part in command
                ),
                "digitiser_repo": str(repo_root),
                "digitiser_model": str(model_dir),
                "digitiser_input_png": str(png_input),
            }
            return None, metadata

        metadata: Dict[str, Any] = {
            "success": False,
            "digitiser_command": " ".join(
                f'"{str(part)}"' if " " in str(part) else str(part)
                for part in command
            ),
            "digitiser_return_code": completed.returncode,
            "digitiser_stdout": completed.stdout,
            "digitiser_stderr": completed.stderr,
            "digitiser_input_png": str(png_input),
            "digitiser_repo": str(repo_root),
            "digitiser_model": str(model_dir),
        }

        if completed.returncode != 0:
            metadata["error"] = (
                "ECG-Digitiser returned a non-zero exit code."
            )
            return None, metadata

        header_files = sorted(output_dir.glob("*.hea"))

        if not header_files:
            metadata["error"] = (
                "ECG-Digitiser completed, but no .hea WFDB file "
                "was produced."
            )

            return None, metadata

        header_path = header_files[0]

        metadata["success"] = True
        metadata["wfdb_header"] = str(header_path)

        return header_path, metadata

    except Exception as exc:
        return None, {
            "success": False,
            "error": f"ECG-Digitiser execution failed: {exc}",
        }

    finally:
        # The WFDB file is read before this temporary directory is removed.
        pass


# ---------------------------------------------------------------------------
# WFDB loading
# ---------------------------------------------------------------------------

def _load_wfdb_record(
    header_path: Path,
) -> Tuple[np.ndarray, list[str], float]:
    """
    Read the WFDB record generated by ECG-Digitiser.
    """

    record_path = header_path.with_suffix("")

    try:
        signal, fields = wfdb.rdsamp(str(record_path))
    except Exception as exc:
        raise RuntimeError(
            f"Could not read ECG-Digitiser WFDB output: "
            f"{record_path}"
        ) from exc

    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 2:
        raise ValueError(
            f"Unexpected WFDB signal shape: {signal.shape}"
        )

    signal_names = [
        str(name).strip()
        for name in fields.get("sig_name", [])
    ]

    if not signal_names:
        raise ValueError(
            "WFDB output does not contain signal names."
        )

    fs = float(fields.get("fs", DIGITISER_FS))

    return signal, signal_names, fs


# ---------------------------------------------------------------------------
# Lead handling
# ---------------------------------------------------------------------------

def _find_lead_index(
    signal_names: list[str],
    target_lead: str,
) -> Optional[int]:
    """
    Find a lead name case-insensitively.
    """

    target = target_lead.strip().lower()

    for index, name in enumerate(signal_names):
        if name.strip().lower() == target:
            return index

    return None


def _select_target_leads(
    signal: np.ndarray,
    signal_names: list[str],
) -> np.ndarray:
    """
    Select exactly the 8 leads required by CardioAgent.

    Required order:

        I, II, V1, V2, V3, V4, V5, V6

    Missing leads are treated as an error.

    ── FIX 2 ──
    After name-matching, also verify each selected lead actually contains
    a non-degenerate signal (not flat, not mostly zeros). This catches the
    case where the digitiser emits a lead name but its mask failed, which
    produces a NaN-padded -> zero-filled column.
    """

    indices = []
    missing = []

    for lead in TARGET_LEADS:
        index = _find_lead_index(
            signal_names,
            lead,
        )

        if index is None:
            missing.append(lead)
        else:
            indices.append(index)

    if missing:
        raise ValueError(
            "ECG image digitization produced an incomplete WFDB record. "
            "The required leads exist in the source ECG page, but the "
            "digitizer did not export them. Missing: "
            + ", ".join(missing)
            + f". Available WFDB leads: {signal_names}"
        )

    selected = signal[:, indices]

    if selected.shape[1] != len(TARGET_LEADS):
        raise ValueError(
            "Unexpected number of selected ECG leads: "
            f"{selected.shape}"
        )

    # ── FIX 2 (content check) ──
    degenerate = []
    for i, lead in enumerate(TARGET_LEADS):
        col = selected[:, i].astype(np.float64)
        zero_frac = float(np.mean(np.isclose(col, 0.0, atol=1e-6)))
        if zero_frac > 0.05 or float(np.nanstd(col)) < 1e-4:
            degenerate.append(
                f"{lead}(zero={zero_frac:.0%},std={float(np.nanstd(col)):.2e})"
            )

    if degenerate:
        raise ValueError(
            "Digitiser produced flat or zero-filled leads: "
            + ", ".join(degenerate)
            + f". Available WFDB leads: {signal_names}"
        )

    return selected.astype(np.float32)


# ---------------------------------------------------------------------------
# Sampling-rate conversion
# ---------------------------------------------------------------------------

def _resample_signal(
    signal: np.ndarray,
    source_fs: float,
    target_fs: float,
) -> np.ndarray:
    """
    Resample the digitizer output to the CardioAgent model sampling rate.

    ECG-Digitiser outputs 500 Hz according to config.py.
    CardioAgent models use 100 Hz.
    """

    if source_fs <= 0:
        raise ValueError(
            f"Invalid source sampling rate: {source_fs}"
        )

    if target_fs <= 0:
        raise ValueError(
            f"Invalid target sampling rate: {target_fs}"
        )

    if np.isclose(source_fs, target_fs):
        return signal.astype(np.float32)

    from fractions import Fraction

    ratio = Fraction(
        float(target_fs) / float(source_fs)
    ).limit_denominator(1000)

    up = ratio.numerator
    down = ratio.denominator

    resampled = resample_poly(
        signal,
        up,
        down,
        axis=0,
    )

    return np.asarray(
        resampled,
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Duration handling
# ---------------------------------------------------------------------------

def _prepare_model_length(
    signal: np.ndarray,
    target_length: int = TARGET_LENGTH,
) -> np.ndarray:
    """
    Prepare the digitized signal for the fixed-size CardioAgent model.

    ── FIX 3 ──
    Previously this padded a short signal by repeating its last sample,
    which created a large DC step (and corrupted per-lead z-scoring in
    preprocess_signal). Now we pad with the per-lead mean, which is a
    neutral value that does not introduce a discontinuity.
    """

    if signal.ndim != 2:
        raise ValueError(
            f"Expected 2D signal, got {signal.shape}"
        )

    current_length = signal.shape[0]

    if current_length == target_length:
        return signal.astype(np.float32)

    if current_length > target_length:
        return signal[:target_length].astype(np.float32)

    if current_length <= 0:
        raise ValueError(
            "Digitized ECG contains no samples."
        )

    pad_length = target_length - current_length

    # ── FIX 3 (mean padding instead of last-sample padding) ──
    per_lead_mean = signal.mean(axis=0, keepdims=True)
    padding = np.repeat(per_lead_mean, pad_length, axis=0)

    return np.concatenate(
        [signal, padding],
        axis=0,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_signal_from_image(
    image_path: str | Path,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Convert an ECG image into a CardioAgent-ready ECG signal.

    Pipeline:

        ECG image
            ↓
        PNG conversion
            ↓
        official ECG-Digitiser
            ↓
        WFDB
            ↓
        validate raw signal  ← FIX 1
            ↓
        select I, II, V1-V6 (with content check)  ← FIX 2
            ↓
        resample 500 Hz -> 100 Hz
            ↓
        prepare 1000 samples (mean padding)  ← FIX 3
            ↓
        CardioAgent ECG model

    Returns a dictionary containing the signal and metadata.
    """

    image_path = Path(image_path).expanduser().resolve()

    if not image_path.exists():
        return {
            "success": False,
            "signal": None,
            "error": f"ECG image not found: {image_path}",
        }

    force_redigitize = os.getenv("ECG_FORCE_REDIGITIZE", "0") == "1"

    if use_cache and not force_redigitize:
        cached_result = _load_cached_signal(image_path)
        if cached_result is not None:
            return cached_result

    header_path = None
    digitiser_metadata: Dict[str, Any] = {}

    temp_root: Optional[Path] = None

    try:
        repo_root = _find_external_digitiser_repo()

        if repo_root is None:
            return {
                "success": False,
                "signal": None,
                "error": (
                    "ECG-Digitiser repository not found."
                ),
            }

        model_dir = _find_digitiser_model(repo_root)

        if model_dir is None:
            return {
                "success": False,
                "signal": None,
                "error": (
                    "ECG-Digitiser M3 model not found."
                ),
            }

        digitize_script = (
            repo_root / "src" / "run" / "digitize.py"
        )

        if not digitize_script.exists():
            return {
                "success": False,
                "signal": None,
                "error": (
                    f"ECG-Digitiser script not found: "
                    f"{digitize_script}"
                ),
            }

        temp_root = Path(
            tempfile.mkdtemp(
                prefix="cardioagent_digitiser_"
            )
        )

        input_dir = temp_root / "input"
        output_dir = temp_root / "output"

        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        png_input = _prepare_png_input(
            image_path,
            input_dir,
        )

        command = [
            sys.executable,
            "-m",
            "src.run.digitize",
            "-d",
            str(input_dir),
            "-m",
            str(model_dir),
            "-o",
            str(output_dir),
            "-v",
        ]

        env = os.environ.copy()

        env["PYTHONPATH"] = os.pathsep.join(
            [
                str(repo_root),
                env.get("PYTHONPATH", ""),
            ]
        ).rstrip(os.pathsep)

        env["PATH"] = os.pathsep.join(
            [
                str(Path(sys.executable).parent),
                env.get("PATH", ""),
            ]
        )

        env["nnUNet_results"] = str(model_dir / "nnUNet_results")

        nnunet_exe = shutil.which("nnUNetv2_predict", path=env.get("PATH"))
        if nnunet_exe is None:
            digitiser_metadata = {
                "success": False,
                "error": (
                    "nnUNetv2_predict was not found in the active Python environment. "
                    "Install the ECG-Digitiser bundled/patched nnUNet package "
                    "before running image digitization."
                ),
                "digitiser_repo": str(repo_root),
                "digitiser_model": str(model_dir),
                "digitiser_input_png": str(png_input),
            }
            return {
                "success": False,
                "signal": None,
                "error": digitiser_metadata["error"],
                "metadata": digitiser_metadata,
            }

        logger.info("Using nnUNetv2_predict: %s", nnunet_exe)
        logger.info(
            "ECG-Digitiser input PNG: %s (%d x %d)",
            png_input,
            Image.open(png_input).size[0],
            Image.open(png_input).size[1],
        )

        try:
            completed = subprocess.run(
                command,
                cwd=str(repo_root),
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=DIGITISER_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            digitiser_metadata = {
                "success": False,
                "error": (
                    f"ECG-Digitiser timed out after {DIGITISER_TIMEOUT_SECONDS} seconds. "
                    "This usually means the model is large, the input is invalid, or the external digitizer is hanging."
                ),
                "digitiser_command": " ".join(
                    f'"{str(part)}"' if " " in str(part) else str(part)
                    for part in command
                ),
                "digitiser_repo": str(repo_root),
                "digitiser_model": str(model_dir),
                "digitiser_input_png": str(png_input),
            }
            return {
                "success": False,
                "signal": None,
                "error": digitiser_metadata["error"],
                "metadata": digitiser_metadata,
            }

        digitiser_metadata = {
            "success": False,
            "digitiser_command": " ".join(
                f'"{str(part)}"' if " " in str(part) else str(part)
                for part in command
            ),
            "digitiser_return_code": completed.returncode,
            "digitiser_stdout": completed.stdout,
            "digitiser_stderr": completed.stderr,
            "digitiser_input_png": str(png_input),
            "digitiser_repo": str(repo_root),
            "digitiser_model": str(model_dir),
        }

        if completed.returncode != 0:
            stderr_tail = (completed.stderr or "").strip()[-2000:]
            stdout_tail = (completed.stdout or "").strip()[-1000:]
            digitiser_metadata["error"] = (
                "ECG-Digitiser returned a non-zero exit code "
                f"({completed.returncode}).\n"
                f"STDERR:\n{stderr_tail}\n"
                f"STDOUT:\n{stdout_tail}"
            )

            return {
                "success": False,
                "signal": None,
                "error": digitiser_metadata["error"],
                "metadata": digitiser_metadata,
            }

        header_files = sorted(
            output_dir.glob("*.hea")
        )

        if not header_files:
            digitiser_metadata["error"] = (
                "ECG-Digitiser completed, but no .hea WFDB "
                "file was produced."
            )

            return {
                "success": False,
                "signal": None,
                "error": digitiser_metadata["error"],
                "metadata": digitiser_metadata,
            }

        header_path = header_files[0]

        digitiser_metadata["success"] = True
        digitiser_metadata["wfdb_header"] = str(
            header_path
        )

        # ---------------------------------------------------------------
        # Read WFDB
        # ---------------------------------------------------------------

        raw_signal, signal_names, source_fs = (
            _load_wfdb_record(header_path)
        )

        digitiser_metadata["raw_signal_shape"] = tuple(
            raw_signal.shape
        )
        digitiser_metadata["raw_signal_names"] = signal_names
        digitiser_metadata["raw_sampling_rate"] = source_fs

        # ── FIX 1: validate raw signal before we trust it ──
        reason = _validate_raw_signal(raw_signal, signal_names, source_fs)
        if reason:
            digitiser_metadata["validation_error"] = reason
            return {
                "success": False,
                "signal": None,
                "error": f"Digitiser produced an invalid signal: {reason}",
                "metadata": digitiser_metadata,
            }

        # ---------------------------------------------------------------
        # Select exact 8 model leads (with content check)
        # ---------------------------------------------------------------

        try:
            selected_signal = _select_target_leads(
                raw_signal,
                signal_names,
            )
        except ValueError as exc:
            digitiser_metadata["lead_selection_error"] = str(exc)
            return {
                "success": False,
                "signal": None,
                "error": str(exc),
                "metadata": digitiser_metadata,
            }

        digitiser_metadata["selected_leads"] = list(
            TARGET_LEADS
        )

        # ---------------------------------------------------------------
        # Resample to CardioAgent model sampling rate
        # ---------------------------------------------------------------

        signal_100hz = _resample_signal(
            selected_signal,
            source_fs=source_fs,
            target_fs=TARGET_FS,
        )

        digitiser_metadata["model_sampling_rate"] = TARGET_FS
        digitiser_metadata["resampled_shape"] = tuple(
            signal_100hz.shape
        )

        # ---------------------------------------------------------------
        # Prepare fixed model length (mean padding)
        # ---------------------------------------------------------------

        model_signal = _prepare_model_length(
            signal_100hz,
            target_length=TARGET_LENGTH,
        )

        digitiser_metadata["model_signal_shape"] = tuple(
            model_signal.shape
        )

        # ---------------------------------------------------------------
        # Basic validity checks
        # ---------------------------------------------------------------

        if not np.isfinite(model_signal).all():
            return {
                "success": False,
                "signal": None,
                "error": (
                    "Digitized ECG contains NaN or infinite values."
                ),
                "metadata": digitiser_metadata,
            }

        if model_signal.shape != (
            TARGET_LENGTH,
            len(TARGET_LEADS),
        ):
            return {
                "success": False,
                "signal": None,
                "error": (
                    "Unexpected final ECG signal shape: "
                    f"{model_signal.shape}. "
                    f"Expected "
                    f"({TARGET_LENGTH}, {len(TARGET_LEADS)})."
                ),
                "metadata": digitiser_metadata,
            }

        # ---------------------------------------------------------------
        # Optional ECG feature extraction
        # ---------------------------------------------------------------

        features: Dict[str, Any] = {}

        if extract_ecg_features is not None:
            try:
                features = extract_ecg_features(
                    model_signal,
                    fs=TARGET_FS,
                )
            except TypeError:
                try:
                    features = extract_ecg_features(
                        model_signal,
                        TARGET_FS,
                    )
                except Exception:
                    features = {}
            except Exception:
                features = {}

        if use_cache:
            try:
                _save_cached_signal(image_path, model_signal, features)
                digitiser_metadata["cache_path"] = str(_cache_path(image_path))
            except (OSError, TypeError, ValueError) as exc:
                logger.warning("Could not cache ECG signal for %s: %s", image_path, exc)

        # ---------------------------------------------------------------
        # Success
        # ---------------------------------------------------------------

        return {
            "success": True,
            "signal": model_signal,
            "leads": list(TARGET_LEADS),
            "sampling_rate": TARGET_FS,
            "duration_seconds": TARGET_DURATION_SECONDS,
            "features": features,
            "metadata": digitiser_metadata,
        }

    except Exception as exc:
        return {
            "success": False,
            "signal": None,
            "error": str(exc),
            "metadata": digitiser_metadata,
        }

    finally:
        if temp_root is not None:
            shutil.rmtree(
                temp_root,
                ignore_errors=True,
            )


# ---------------------------------------------------------------------------
# Backward-compatible function name
# ---------------------------------------------------------------------------

def image_to_signal(
    image_path: str | Path,
) -> Dict[str, Any]:
    """
    Backward-compatible wrapper.
    """

    return extract_signal_from_image(image_path)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test CardioAgent ECG image digitization."
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to an ECG image.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("CARDIOAGENT ECG IMAGE -> SIGNAL TEST")
    print("=" * 70)
    print(f"Image: {args.image}")
    print()

    result = extract_signal_from_image(
        args.image
    )

    if not result.get("success"):
        print("FAILED")
        print()
        print("Error:")
        print(result.get("error"))

        metadata = result.get("metadata", {})

        if metadata:
            print()
            print("Metadata:")

            for key, value in metadata.items():
                print(f"  {key}: {value}")

        raise SystemExit(1)

    signal = result["signal"]

    print("SUCCESS")
    print()
    print(f"Signal shape: {signal.shape}")
    print(f"Leads: {result['leads']}")
    print(f"Sampling rate: {result['sampling_rate']} Hz")
    print(
        f"Duration: {result['duration_seconds']} seconds"
    )
    print()
    print("Lead order:")
    print("  " + " | ".join(result["leads"]))
    print()
    print("First 5 samples:")
    print(signal[:5])
    print()
    print("=" * 70)