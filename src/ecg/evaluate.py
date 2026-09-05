"""
evaluate.py

Evaluates the custom CNN and both baselines on the PTB-XL test split
(strat_fold == 10) and writes results to:

    docs/evaluation/ecg_model_metrics.json

Metrics computed per model:
  - Per-class Precision / Recall / F1
  - Macro-F1
  - Macro-AUROC
  - AUPRC (per-class + macro)
  - Confusion matrix
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.ecg.dataset import IDX_TO_LABEL, PTBXLDataset
from src.ecg.models.cnn1d import CNN1D
from src.ecg.models.inception1d import Inception1D
from src.ecg.models.resnet1d import ResNet1D
from src.ecg.train import get_device

MODEL_REGISTRY = {
    "cnn1d": CNN1D,
    "resnet1d": ResNet1D,
    "inception1d": Inception1D,
}


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for signals, labels in loader:
        signals = signals.to(device)
        logits = model(signals)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray) -> dict:
    """
    Args:
        y_true: (n,) integer class labels
        y_probs: (n, n_classes) predicted probabilities

    Returns:
        dict of metrics, JSON-serializable.
    """
    y_pred = y_probs.argmax(axis=1)
    n_classes = y_probs.shape[1]
    class_names = [IDX_TO_LABEL[i] for i in range(n_classes)]

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # one-hot for multi-class AUROC/AUPRC
    y_true_onehot = np.eye(n_classes)[y_true]
    try:
        macro_auroc = roc_auc_score(
            y_true_onehot, y_probs, average="macro", multi_class="ovr"
        )
    except ValueError:
        macro_auroc = None  # can fail if a class is absent from y_true

    per_class_auprc = {}
    for i, name in enumerate(class_names):
        per_class_auprc[name] = float(
            average_precision_score(y_true_onehot[:, i], y_probs[:, i])
        )
    macro_auprc = float(np.mean(list(per_class_auprc.values())))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    return {
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in class_names
        },
        "macro_f1": float(macro_f1),
        "macro_auroc": macro_auroc,
        "auprc": {"per_class": per_class_auprc, "macro": macro_auprc},
        "confusion_matrix": {
            "labels": class_names,
            "matrix": cm.tolist(),
        },
        "accuracy": float(report["accuracy"]),
    }


def evaluate_all_models(
    ptbxl_root: str,
    checkpoint_dir: str = "checkpoints",
    output_path: str = "docs/evaluation/ecg_model_metrics.json",
    sampling_rate: int = 100,
    batch_size: int = 64,
):
    device = get_device()
    test_ds = PTBXLDataset(ptbxl_root, split="test", sampling_rate=sampling_rate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    results = {}
    for model_name, model_cls in MODEL_REGISTRY.items():
        ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"Skipping {model_name}: no checkpoint found at {ckpt_path}")
            continue

        model = model_cls(
            in_channels=test_ds.input_channels, num_classes=test_ds.num_classes
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        y_probs, y_true = get_predictions(model, test_loader, device)
        metrics = compute_metrics(y_true, y_probs)
        results[model_name] = metrics
        print(f"{model_name}: macro_f1={metrics['macro_f1']:.4f} "
              f"macro_auroc={metrics['macro_auroc']}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote metrics to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl-root", default=os.environ.get("PTBXL_ROOT", "data/raw/ptbxl"))
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output", default="docs/evaluation/ecg_model_metrics.json")
    parser.add_argument("--sampling-rate", type=int, default=100, choices=[100, 500])
    args = parser.parse_args()

    evaluate_all_models(
        ptbxl_root=args.ptbxl_root,
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output,
        sampling_rate=args.sampling_rate,
    )


if __name__ == "__main__":
    main()
