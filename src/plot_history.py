"""
plot_history.py
----------------
Plots train/val loss and accuracy curves from a history JSON saved by train.py.
Useful for the report -- shows convergence and whether the model overfits.

Usage:
    python plot_history.py --history ../outputs/history_alexnet.json
"""

import argparse
import json

import matplotlib.pyplot as plt


def plot_history(history_path: str, output_path: str = "training_curves.png"):
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Accuracy")
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training history curves.")
    parser.add_argument("--history", required=True, help="Path to history JSON from train.py")
    parser.add_argument("--output", default="training_curves.png")
    args = parser.parse_args()

    plot_history(args.history, args.output)
