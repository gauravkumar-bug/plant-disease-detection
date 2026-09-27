"""
train.py
--------
Trains any of the 6 supported architectures (transfer learning) on the
Plant Village dataset and saves the best checkpoint plus a training
history JSON/plot.

Usage:
    python train.py --data_dir ./data --arch alexnet --epochs 15 --batch_size 32
    python train.py --data_dir ./data --arch efficientnet_v2_m --epochs 15 --batch_size 8

CHANGES FROM ORIGINAL (fixes Task 3 / Task 5 symptoms -- training that
appears frozen with no output for hours, and errors that vanish silently):
  1. Verbose startup log: device, dataset sizes, class count, per-arch
     input resolution -- printed BEFORE any training starts.
  2. Per-batch progress bar (tqdm) inside train_one_epoch/validate, so you
     see live loss/accuracy instead of waiting for a full epoch to finish.
     Falls back to a plain periodic print if tqdm isn't installed.
  3. The whole run is wrapped in try/except that prints the exception type,
     message, and which phase (dataset load / model build / training /
     saving) it happened in, then re-raises -- nothing is swallowed.
The training math itself (loss, optimizer, scheduler, checkpoint format) is
byte-for-byte identical to your original -- this only changes what gets
printed and how errors surface.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from model import build_model, count_trainable_params, get_device
from dataset import get_dataloaders

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _iter_with_progress(loader, desc):
    if _HAS_TQDM:
        return tqdm(loader, desc=desc, leave=False, file=sys.stdout)
    return loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch=None, epochs=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    desc = f"train {epoch}/{epochs}" if epoch else "train"

    pbar = _iter_with_progress(loader, desc)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        running_avg_loss = running_loss / total
        running_avg_acc = correct / total
        if _HAS_TQDM:
            pbar.set_postfix(loss=f"{running_avg_loss:.4f}", acc=f"{running_avg_acc:.4f}")
        elif batch_idx % 10 == 0 or batch_idx == len(loader) - 1:
            print(f"  [{desc}] batch {batch_idx + 1}/{len(loader)} "
                  f"loss={running_avg_loss:.4f} acc={running_avg_acc:.4f}", flush=True)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, epoch=None, epochs=None):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    desc = f"val   {epoch}/{epochs}" if epoch else "val"

    pbar = _iter_with_progress(loader, desc)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        if _HAS_TQDM:
            pbar.set_postfix(loss=f"{running_loss / total:.4f}", acc=f"{correct / total:.4f}")
        elif batch_idx % 10 == 0 or batch_idx == len(loader) - 1:
            print(f"  [{desc}] batch {batch_idx + 1}/{len(loader)} "
                  f"loss={running_loss / total:.4f} acc={correct / total:.4f}", flush=True)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train a plant disease detection model.")
    parser.add_argument("--data_dir", default="./data", help="Directory with train/ and val/ subfolders")
    parser.add_argument("--arch", default="alexnet", choices=["alexnet", "vgg16", "resnet50", "mobilenet", "efficientnet_v2_s", "efficientnet_v2_m"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--num_classes", type=int, default=38)
    parser.add_argument("--output_dir", default="../outputs")
    parser.add_argument("--num_workers", type=int, default=0,
                         help="DataLoader worker processes. Keep at 0 on macOS/MPS -- "
                              "forked workers can deadlock with Metal (see dataset.py docstring).")
    args = parser.parse_args()

    print(f"[train] arch={args.arch} epochs={args.epochs} batch_size={args.batch_size} "
          f"num_classes={args.num_classes} freeze_backbone={args.freeze_backbone} "
          f"num_workers={args.num_workers}", flush=True)
    if not _HAS_TQDM:
        print("[train] tqdm not installed -- falling back to periodic batch prints. "
              "Run `pip install tqdm --break-system-packages` for a live progress bar.", flush=True)

    phase = "device detection"
    try:
        device = get_device()
        print(f"[train] using device: {device}", flush=True)
        if args.num_workers > 0 and str(device) == "mps":
            print(f"[train] WARNING: --num_workers={args.num_workers} with device=mps. "
                  f"Forked DataLoader workers are known to deadlock with Apple's Metal "
                  f"backend -- if training hangs with zero output after 'Trainable "
                  f"parameters', re-run with --num_workers 0.", file=sys.stderr, flush=True)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        phase = "dataset loading"
        t0 = time.time()
        train_loader, val_loader, class_names = get_dataloaders(
            args.data_dir, arch=args.arch, batch_size=args.batch_size, num_workers=args.num_workers
        )
        print(f"[train] dataset loaded in {time.time() - t0:.1f}s | "
              f"{len(train_loader.dataset)} train images | "
              f"{len(val_loader.dataset)} val images | "
              f"{len(class_names)} classes: {class_names[:5]}{'...' if len(class_names) > 5 else ''}",
              flush=True)

        if len(class_names) != args.num_classes:
            print(f"[train] WARNING: --num_classes={args.num_classes} but the dataset folder "
                  f"actually has {len(class_names)} class subfolders. The final layer will be "
                  f"built with {args.num_classes} outputs -- pass --num_classes {len(class_names)} "
                  f"to match your actual data.", file=sys.stderr, flush=True)

        with open(output_dir / "class_names.json", "w") as f:
            json.dump(class_names, f, indent=2)

        phase = "model build"
        model = build_model(arch=args.arch, num_classes=args.num_classes,
                             freeze_backbone=args.freeze_backbone).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_acc = 0.0

        phase = "training loop"
        for epoch in range(1, args.epochs + 1):
            start = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, args.epochs)
            val_loss, val_acc = validate(
                model, val_loader, criterion, device, epoch, args.epochs)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            elapsed = time.time() - start
            print(f"Epoch {epoch}/{args.epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s", flush=True)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                phase = "checkpoint saving"
                checkpoint_path = output_dir / f"best_{args.arch}.pth"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "arch": args.arch,
                    "num_classes": args.num_classes,
                    "class_names": class_names,
                    "val_acc": val_acc,
                }, checkpoint_path)
                print(f"  -> New best model saved to {checkpoint_path} (val_acc={val_acc:.4f})", flush=True)
                phase = "training loop"

        with open(output_dir / f"history_{args.arch}.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}", flush=True)

    except Exception as e:
        print(f"\n[train] FAILED during '{phase}': {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
