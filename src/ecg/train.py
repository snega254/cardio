"""
train.py

Trains the custom CNN and both published baselines (ResNet1D,
Inception1D) on the same PTB-XL strat_fold split, saving each model's
weights so evaluate.py can load and score them.

Usage:
    python -m src.ecg.train --model cnn1d --epochs 30
    python -m src.ecg.train --model resnet1d --epochs 30
    python -m src.ecg.train --model inception1d --epochs 30
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.ecg.dataset import PTBXLDataset
from src.ecg.models.cnn1d import CNN1D
from src.ecg.models.inception1d import Inception1D
from src.ecg.models.resnet1d import ResNet1D

MODEL_REGISTRY = {
    "cnn1d": CNN1D,
    "resnet1d": ResNet1D,
    "inception1d": Inception1D,
}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * signals.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loss_acc(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        logits = model(signals)
        loss = criterion(logits, labels)
        total_loss += loss.item() * signals.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def train_model(
    model_name: str,
    ptbxl_root: str,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    sampling_rate: int = 100,
    checkpoint_dir: str = "checkpoints",
):
    device = get_device()
    print(f"Training {model_name} on {device}")

    train_ds = PTBXLDataset(ptbxl_root, split="train", sampling_rate=sampling_rate)
    val_ds = PTBXLDataset(ptbxl_root, split="val", sampling_rate=sampling_rate)

    # num_workers=0 avoids Windows spawn-based multiprocessing pickling
    # errors when the Dataset holds large in-memory arrays (eager-loaded
    # in PTBXLDataset.__init__). Safe default for CPU-only training.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(
        in_channels=train_ds.input_channels, num_classes=train_ds.num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    # weight_decay adds L2 regularization to combat overfitting
    # (observed as val_loss rising while train_loss kept falling).
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_loss_acc(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"[{model_name}] epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  -> saved new best checkpoint to {best_path}")

    return best_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    parser.add_argument("--ptbxl-root", default=os.environ.get("PTBXL_ROOT", "data/raw/ptbxl"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sampling-rate", type=int, default=100, choices=[100, 500])
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        ptbxl_root=args.ptbxl_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sampling_rate=args.sampling_rate,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()