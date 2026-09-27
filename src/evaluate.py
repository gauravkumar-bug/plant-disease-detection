"""
evaluate.py
-----------
Loads a trained checkpoint and reports detailed metrics on the validation
set: accuracy, per-class precision/recall/F1, and a confusion matrix plot.
This goes beyond raw accuracy, which is worth having ready if an interviewer
asks "accuracy alone isn't enough -- what about per-class performance?"

Usage:
    python evaluate.py --checkpoint ../outputs/best_alexnet.pth --data_dir ./data
"""

import argparse
import json

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from model import build_model, get_device
from dataset import get_dataloaders


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained plant disease model.")
    parser.add_argument("--checkpoint", required=True, help="Path to saved .pth checkpoint")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = get_device()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    arch = checkpoint["arch"]
    class_names = checkpoint["class_names"]

    model = build_model(arch=arch, num_classes=checkpoint["num_classes"], freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    _, val_loader, _ = get_dataloaders(args.data_dir, arch=arch, batch_size=args.batch_size)

    preds, labels = get_predictions(model, val_loader, device)

    print(f"\nCheckpoint val_acc at save time: {checkpoint['val_acc']:.4f}\n")
    print("Classification report:")
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(labels, preds)
    with open("confusion_matrix.json", "w") as f:
        json.dump(cm.tolist(), f)

    # Save a heatmap image (readable even with 38 classes at larger figure size)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {arch}")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)
    print("\nSaved confusion_matrix.png and confusion_matrix.json")


if __name__ == "__main__":
    main()
