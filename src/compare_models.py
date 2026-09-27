"""
compare_models.py
------------------
Runs a short training pass for each supported architecture on the same
dataset and produces bar-chart comparisons of:
  - final validation accuracy
  - final validation loss
  - total training time
  - number of trainable parameters

Usage:
    python compare_models.py --data_dir ./data --epochs 5 --num_classes 15 --num_workers 0 --output_dir ./outputs

CHANGES FROM ORIGINAL (this matters for a multi-hour unattended run):
  1. Results are written to comparison_results.json after EVERY architecture
     finishes, not just once at the very end. If the run is interrupted
     (crash, laptop sleep, accidental Ctrl+C, terminal mishap) partway
     through, you keep every result up to that point instead of losing all
     of them.
  2. RESUMABLE: if comparison_results.json already has an entry for an
     arch, that arch is skipped on re-run. So if it dies on arch 5/6, you
     just re-run the exact same command and it picks up from arch 5,
     it does not redo 1-4.
  3. Each arch's training is wrapped in try/except -- if one architecture
     fails (OOM, bad weights download, etc.), the error is logged and the
     script moves on to the next arch instead of the whole comparison run
     dying.
  4. Fixed a real bug: the color list only had 4 entries hardcoded, but
     SUPPORTED_ARCHS has 6 -- plotting all 6 would have broken or
     mis-colored the chart. Now generates enough colors for any number
     of archs.
  5. --num_workers exposed (default 0 -- see dataset.py docstring re:
     macOS + MPS fork deadlocks).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from model import build_model, count_trainable_params, get_device, SUPPORTED_ARCHS
from dataset import get_dataloaders
from train import train_one_epoch, validate


def load_existing_results(output_dir: Path) -> dict:
    results_path = output_dir / "comparison_results.json"
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
        print(f"[compare] found existing results for: {list(existing.keys())} "
              f"-- these will be SKIPPED (delete comparison_results.json to redo them)", flush=True)
        return existing
    return {}


def save_results(results: dict, output_dir: Path):
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)


def run_comparison(data_dir: str, archs, epochs: int, batch_size: int,
                    num_classes: int, lr: float, output_dir: str, num_workers: int = 0):
    device = get_device()
    print(f"[compare] using device: {device}", flush=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_existing_results(output_dir)
    remaining = [a for a in archs if a not in results]
    print(f"[compare] {len(remaining)}/{len(archs)} architectures left to run: {remaining}", flush=True)

    overall_start = time.time()
    for idx, arch in enumerate(remaining, 1):
        print(f"\n{'=' * 50}\n[compare] arch {idx}/{len(remaining)}: {arch}\n{'=' * 50}", flush=True)

        try:
            t0 = time.time()
            train_loader, val_loader, class_names = get_dataloaders(
                data_dir, arch=arch, batch_size=batch_size, num_workers=num_workers
            )
            print(f"[compare] {arch}: dataset loaded in {time.time() - t0:.1f}s | "
                  f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val images | "
                  f"{len(class_names)} classes", flush=True)
            if len(class_names) != num_classes:
                print(f"[compare] WARNING: --num_classes={num_classes} but the dataset folder "
                      f"actually has {len(class_names)} class subfolders.", flush=True)

            model = build_model(arch=arch, num_classes=num_classes, freeze_backbone=True).to(device)
            trainable_params = count_trainable_params(model)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

            start_time = time.time()
            for epoch in range(1, epochs + 1):
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
                val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, epochs)
                print(f"  [{arch}] epoch {epoch}/{epochs} "
                      f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}", flush=True)
            total_time = time.time() - start_time

            results[arch] = {
                "final_val_acc": val_acc,
                "final_val_loss": val_loss,
                "final_train_acc": train_acc,
                "trainable_params": trainable_params,
                "training_time_sec": total_time,
            }

            torch.save({
                "model_state_dict": model.state_dict(),
                "arch": arch,
                "num_classes": num_classes,
                "class_names": class_names,
                "val_acc": val_acc,
            }, output_dir / f"compare_{arch}.pth")

            # Save immediately -- do not wait for the whole loop to finish.
            save_results(results, output_dir)
            print(f"[compare] {arch} done in {total_time:.1f}s, val_acc={val_acc:.4f}. "
                  f"Results saved to {output_dir}/comparison_results.json ({len(results)}/{len(archs)} complete).",
                  flush=True)

        except Exception as e:
            print(f"[compare] FAILED on '{arch}': {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            print(f"[compare] continuing to next architecture -- '{arch}' will be retried "
                  f"if you re-run this command (its result was not saved).", file=sys.stderr, flush=True)
            continue

    elapsed_total = time.time() - overall_start
    print(f"\n[compare] all requested architectures processed in {elapsed_total/60:.1f} min total.", flush=True)

    if len(results) < len(archs):
        missing = [a for a in archs if a not in results]
        print(f"[compare] WARNING: {len(missing)} architecture(s) never completed successfully: {missing}. "
              f"Chart below only reflects the {len(results)} that succeeded. Re-run the same command to retry them.",
              file=sys.stderr, flush=True)

    if results:
        plot_comparison(results, output_dir)
    print(f"\nResults saved to {output_dir}/comparison_results.json")
    return results


def plot_comparison(results: dict, output_dir: Path):
    """Build a 2x2 grid of bar charts: accuracy, loss, training time, params."""
    archs = list(results.keys())
    val_accs = [results[a]["final_val_acc"] * 100 for a in archs]
    val_losses = [results[a]["final_val_loss"] for a in archs]
    times = [results[a]["training_time_sec"] for a in archs]
    params = [results[a]["trainable_params"] / 1e6 for a in archs]  # in millions

    # Enough distinct colors for any number of archs (was hardcoded to 4 -- broke on 6).
    cmap = plt.colormaps.get("tab10")
    colors = [cmap(i % 10) for i in range(len(archs))]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].bar(archs, val_accs, color=colors)
    axes[0, 0].set_title("Validation Accuracy (%)")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].tick_params(axis="x", rotation=30)
    for i, v in enumerate(val_accs):
        axes[0, 0].text(i, v, f"{v:.1f}%", ha="center", va="bottom")

    axes[0, 1].bar(archs, val_losses, color=colors)
    axes[0, 1].set_title("Validation Loss")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].tick_params(axis="x", rotation=30)
    for i, v in enumerate(val_losses):
        axes[0, 1].text(i, v, f"{v:.3f}", ha="center", va="bottom")

    axes[1, 0].bar(archs, times, color=colors)
    axes[1, 0].set_title("Training Time (seconds)")
    axes[1, 0].set_ylabel("Seconds")
    axes[1, 0].tick_params(axis="x", rotation=30)
    for i, v in enumerate(times):
        axes[1, 0].text(i, v, f"{v:.0f}s", ha="center", va="bottom")

    axes[1, 1].bar(archs, params, color=colors)
    axes[1, 1].set_title("Trainable Parameters (Millions, log scale)")
    axes[1, 1].set_ylabel("Params (M)")
    axes[1, 1].set_yscale("log")
    axes[1, 1].tick_params(axis="x", rotation=30)
    for i, v in enumerate(params):
        label = f"{v:.3f}M" if v < 1 else f"{v:.1f}M"
        axes[1, 1].text(i, v, label, ha="center", va="bottom")

    fig.suptitle(f"Model Comparison: {' vs '.join(archs)}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=150)
    print(f"Saved {output_dir / 'model_comparison.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare architectures from SUPPORTED_ARCHS.")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--archs", nargs="+", default=SUPPORTED_ARCHS,
                         help=f"Which architectures to compare, from {SUPPORTED_ARCHS}")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per model (comparison run)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=38)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="../outputs")
    parser.add_argument("--num_workers", type=int, default=0,
                         help="Keep at 0 on macOS/MPS -- see dataset.py docstring.")
    args = parser.parse_args()

    run_comparison(args.data_dir, args.archs, args.epochs, args.batch_size,
                    args.num_classes, args.lr, args.output_dir, args.num_workers)
