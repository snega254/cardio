"""
ECG Model Configuration
Centralizes model paths and settings.
Synced with interface.py.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Best model (confirmed by user)
BEST_MODEL = "inception1d"

# Checkpoint directory (env-configurable)
CHECKPOINT_DIR = os.getenv("ECG_CHECKPOINT_DIR", "checkpoints")

# Available models
AVAILABLE_MODELS = {
    "inception1d": f"{CHECKPOINT_DIR}/inception1d_best.pt",
    "resnet1d": f"{CHECKPOINT_DIR}/resnet1d_best.pt",
    "cnn1d": f"{CHECKPOINT_DIR}/cnn1d_best.pt",
}

# Fallback order
FALLBACK_ORDER = ["inception1d", "resnet1d", "cnn1d"]


def get_model_path(model_name: str = None) -> str:
    """Get the file path for the specified model."""
    if model_name is None:
        model_name = BEST_MODEL

    env_path = os.getenv("ECG_MODEL_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    if model_name in AVAILABLE_MODELS:
        path = AVAILABLE_MODELS[model_name]
        if Path(path).exists():
            return path

    raise FileNotFoundError(f"Model not found: {model_name}")


def list_available_models() -> dict:
    """List all available models with their status."""
    result = {}
    for name, path in AVAILABLE_MODELS.items():
        p = Path(path)
        result[name] = {
            "path": path,
            "exists": p.exists(),
            "size_mb": round(p.stat().st_size / 1e6, 2) if p.exists() else 0,
            "is_best": name == BEST_MODEL,
        }
    return result


if __name__ == "__main__":
    print("Available ECG Models:")
    print("=" * 60)
    for name, info in list_available_models().items():
        marker = " ⭐ (BEST)" if info["is_best"] else ""
        status = "✅" if info["exists"] else "❌"
        print(f"{status} {name}{marker}")
        print(f"   Path: {info['path']}")
        print(f"   Size: {info['size_mb']} MB")
        print()