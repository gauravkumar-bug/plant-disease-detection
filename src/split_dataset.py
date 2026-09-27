"""
split_dataset.py
-----------------
One-time utility to split the raw Plant Village dataset (downloaded from
Kaggle, one folder per class) into train/ and val/ folders in an 80/20 split,
matching the "Split into 80% training and 20% testing" step in the report.

Usage:
    python split_dataset.py --source /path/to/PlantVillage --dest ./data --val_ratio 0.2
"""

import argparse
import random
import shutil
from pathlib import Path


def split_dataset(source_dir: str, dest_dir: str, val_ratio: float = 0.2, seed: int = 42):
    random.seed(seed)
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    class_folders = [d for d in source_path.iterdir() if d.is_dir()]
    if not class_folders:
        raise ValueError(f"No class subfolders found in {source_dir}. "
                          f"Expected one folder per disease class.")

    for class_folder in class_folders:
        images = list(class_folder.glob("*"))
        images = [f for f in images if f.is_file()]
        random.shuffle(images)

        split_idx = int(len(images) * (1 - val_ratio))
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        train_class_dir = dest_path / "train" / class_folder.name
        val_class_dir = dest_path / "val" / class_folder.name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

        for img in train_images:
            shutil.copy2(img, train_class_dir / img.name)
        for img in val_images:
            shutil.copy2(img, val_class_dir / img.name)

        print(f"{class_folder.name}: {len(train_images)} train / {len(val_images)} val")

    print(f"\nDone. Dataset split into {dest_path / 'train'} and {dest_path / 'val'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Plant Village dataset into train/val folders.")
    parser.add_argument("--source", required=True, help="Path to raw dataset (one folder per class)")
    parser.add_argument("--dest", default="./data", help="Output directory for train/val split")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of data used for validation")
    args = parser.parse_args()

    split_dataset(args.source, args.dest, args.val_ratio)
