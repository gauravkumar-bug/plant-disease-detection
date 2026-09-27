"""
dataset.py
----------
Loads the Plant Village dataset from a folder structure and applies the
pre-processing steps described in the project report:
  - resizing (AlexNet: 227x227, VGG16: 224x224)
  - normalization (pixel values scaled 0-1, then ImageNet mean/std normalized
    since we're using ImageNet-pretrained backbones)
  - data augmentation (flip, rotation) for the training set

Expected folder structure (standard torchvision ImageFolder layout):

    data/
      train/
        Apple___Apple_scab/
          img1.jpg
          img2.jpg
          ...
        Apple___Black_rot/
          ...
        ... (38 class folders total)
      val/
        Apple___Apple_scab/
          ...
        ...

Download the Plant Village dataset from Kaggle:
https://www.kaggle.com/datasets/emmarex/plantdisease
then split it 80/20 into train/ and val/ folders (see split_dataset.py).
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import INPUT_SIZES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(arch: str = "alexnet"):
    """Return (train_transform, val_transform) sized for the chosen architecture."""
    img_size = INPUT_SIZES[arch.lower()]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # Data augmentation, matching the report's "Key Features" section
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),  # also scales pixels to [0, 1]
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_transform


def get_dataloaders(data_dir: str, arch: str = "alexnet", batch_size: int = 32, num_workers: int = 0):
    """
    Build train/val DataLoaders from an ImageFolder-structured dataset directory.

    Args:
        data_dir: path containing train/ and val/ subfolders.
        arch: controls input image size (see model.INPUT_SIZES).
        batch_size: batch size for both loaders.
        num_workers: DataLoader worker processes. Defaults to 0.

        IMPORTANT (macOS + MPS): num_workers > 0 forks worker processes, and
        Apple's Metal/MPS backend is NOT fork-safe. Once the main process has
        touched the MPS device, forked DataLoader workers can deadlock
        silently -- the main process just hangs forever waiting on
        self._data_queue.get(), with no error, no traceback, nothing. This is
        exactly the "runs for hours with no output" symptom. Keep
        num_workers=0 on Mac unless you've confirmed otherwise; it's slower
        per-batch but it's correct and it won't hang.
    """
    train_transform, val_transform = get_transforms(arch)

    train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(f"{data_dir}/val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes
    return train_loader, val_loader, class_names
