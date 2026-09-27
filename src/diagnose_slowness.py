"""
diagnose_slowness.py
---------------------
Isolates WHY efficientnet_v2_s is taking ~14s per 8-image batch, by timing
three things separately instead of guessing:

  1. Pure GPU compute: forward+backward on random in-memory tensors
     (no disk, no dataset, no iCloud involved at all).
  2. Pure disk read: reading N real files from your data/train folder,
     with no model, no transform, nothing but bytes off disk.
  3. Whether PyTorch is silently falling back any MPS ops to CPU
     (this prints loudly if PYTORCH_ENABLE_MPS_FALLBACK triggers).

Run from the project root:
    python3 src/diagnose_slowness.py --data_dir ./data
"""

import argparse
import os
import sys
import time
import warnings

# Surface MPS fallback warnings instead of letting them scroll past silently
warnings.filterwarnings("always", message=".*MPS.*")

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model, get_device, INPUT_SIZES


def time_pure_compute(arch, device, batch_size=8, n_batches=5):
    print(f"\n=== TEST 1: pure GPU compute, no disk, no dataset ===", flush=True)
    model = build_model(arch=arch, num_classes=15, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    size = INPUT_SIZES[arch]

    # warm-up (first call often includes MPS graph compilation -- exclude from timing)
    dummy = torch.randn(batch_size, 3, size, size, device=device)
    labels = torch.randint(0, 15, (batch_size,), device=device)
    out = model(dummy)
    loss = criterion(out, labels)
    loss.backward()
    if device.type == "mps":
        torch.mps.synchronize()

    times = []
    for i in range(n_batches):
        dummy = torch.randn(batch_size, 3, size, size, device=device)
        labels = torch.randint(0, 15, (batch_size,), device=device)
        t0 = time.time()
        optimizer.zero_grad()
        out = model(dummy)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  batch {i+1}/{n_batches}: {elapsed:.2f}s", flush=True)

    avg = sum(times) / len(times)
    print(f"  -> average pure-compute time per batch: {avg:.2f}s", flush=True)
    return avg


def time_pure_disk_read(data_dir, n_files=40):
    print(f"\n=== TEST 2: pure disk read, no model, no GPU ===", flush=True)
    from pathlib import Path
    train_dir = Path(data_dir) / "train"
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        print("  Could not find class subfolders under data/train -- skipping.", flush=True)
        return None

    files = []
    for d in class_dirs:
        files.extend(list(d.glob("*"))[:5])
        if len(files) >= n_files:
            break
    files = files[:n_files]

    t0 = time.time()
    total_bytes = 0
    for f in files:
        with open(f, "rb") as fh:
            total_bytes += len(fh.read())
    elapsed = time.time() - t0

    per_file = elapsed / len(files) if files else 0
    print(f"  read {len(files)} files ({total_bytes/1024:.0f} KB) in {elapsed:.2f}s "
          f"({per_file*1000:.0f}ms/file)", flush=True)
    if per_file > 0.3:
        print(f"  -> SLOW disk reads ({per_file*1000:.0f}ms/file). This strongly suggests the "
              f"files are iCloud placeholders being downloaded on access, not local disk reads. "
              f"Check System Settings > [your name] > iCloud > iCloud Drive > "
              f"'Desktop & Documents Folders' -- if it's ON, that's very likely the actual cause.",
              flush=True)
    else:
        print(f"  -> disk reads are fast, local files are not the bottleneck.", flush=True)
    return per_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--arch", default="efficientnet_v2_s")
    args = parser.parse_args()

    device = get_device()
    print(f"[diagnose] device: {device}", flush=True)
    print(f"[diagnose] torch version: {torch.__version__}", flush=True)
    print(f"[diagnose] PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '(not set)')}",
          flush=True)

    compute_time = time_pure_compute(args.arch, device)
    disk_time = time_pure_disk_read(args.data_dir)

    print(f"\n=== VERDICT ===", flush=True)
    print(f"Pure GPU compute per batch of 8: {compute_time:.2f}s", flush=True)
    if compute_time > 2.0:
        print("-> The GPU compute itself is slow, independent of any data loading. "
              "This points to MPS operator fallback (check the warnings printed above this line "
              "for 'not currently supported on the MPS backend, will fall back to CPU') "
              "or genuine hardware limits for this architecture on your machine.", flush=True)
    else:
        print("-> GPU compute is fast on its own. If your real training run is still ~14s/batch, "
              "the bottleneck is in data loading/disk (see TEST 2 above), not the model.", flush=True)


if __name__ == "__main__":
    main()
