"""
replot_comparison.py
---------------------
Regenerates outputs/model_comparison.png from an already-saved
outputs/comparison_results.json -- no training involved. Use this whenever
compare_models.py finishes training successfully but the plotting step
itself fails (e.g. a matplotlib API mismatch), so you never have to
retrain anything just to get the chart.

Usage:
    python3 src/replot_comparison.py --output_dir ./outputs
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_comparison(results: dict, output_dir: Path):
    archs = list(results.keys())
    val_accs = [results[a]["final_val_acc"] * 100 for a in archs]
    val_losses = [results[a]["final_val_loss"] for a in archs]
    times = [results[a]["training_time_sec"] for a in archs]
    params = [results[a]["trainable_params"] / 1e6 for a in archs]

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="../outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results_path = output_dir / "comparison_results.json"
    if not results_path.exists():
        raise SystemExit(f"No comparison_results.json found at {results_path}")

    with open(results_path) as f:
        results = json.load(f)

    print(f"Loaded results for: {list(results.keys())}")
    plot_comparison(results, output_dir)
