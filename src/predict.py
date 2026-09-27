"""
predict.py
----------
Runs inference on a single leaf image and prints the predicted disease
class + confidence -- this is the "deployment" piece: a farmer/user
uploads one photo and gets a prediction back.

Usage:
    python predict.py --checkpoint ../outputs/best_alexnet.pth --image ./sample_leaf.jpg
"""

import argparse

import torch
from PIL import Image

from model import build_model, get_device
from dataset import get_transforms


def predict_image(checkpoint_path: str, image_path: str, top_k: int = 3):
    device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    arch = checkpoint["arch"]
    class_names = checkpoint["class_names"]

    model = build_model(arch=arch, num_classes=checkpoint["num_classes"], freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _, val_transform = get_transforms(arch)
    image = Image.open(image_path).convert("RGB")
    input_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        top_probs, top_indices = torch.topk(probabilities, top_k)

    print(f"\nPredictions for {image_path}:")
    for prob, idx in zip(top_probs, top_indices):
        print(f"  {class_names[idx]:40s} {prob.item() * 100:.2f}%")

    return class_names[top_indices[0]], top_probs[0].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict plant disease from a leaf image.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained .pth checkpoint")
    parser.add_argument("--image", required=True, help="Path to a leaf image (jpg/png)")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    predict_image(args.checkpoint, args.image, args.top_k)
