"""
model.py
--------
Builds AlexNet, VGG16, ResNet50, MobileNetV2, EfficientNetV2-S, or
EfficientNetV2-M with ImageNet-pretrained weights and replaces the final
classification layer for plant disease detection.

CHANGE FROM ORIGINAL: build_model() now prints a visible notice before/after
fetching pretrained weights. This was previously silent -- on a slow/flaky
connection, or the first time efficientnet_v2_s/m are built (their weight
files haven't been cached locally yet, unlike the other 4 archs you already
trained before), construction could sit there downloading for a long time
with zero console output, which is indistinguishable from a hang. Now you'll
see exactly which phase you're in.
"""

import sys
import time

import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHS = ["alexnet", "vgg16", "resnet50", "mobilenet", "efficientnet_v2_s", "efficientnet_v2_m"]

INPUT_SIZES = {
    "alexnet": 227,
    "vgg16": 224,
    "resnet50": 224,
    "mobilenet": 224,
    "efficientnet_v2_s": 384,
    "efficientnet_v2_m": 480,
}

# Rough published GFLOPs/image at each arch's native resolution, only used
# to print an honest heads-up for the two heaviest models -- not used in
# any computation.
_APPROX_GFLOPS = {
    "alexnet": 0.7,
    "vgg16": 15.5,
    "resnet50": 4.1,
    "mobilenet": 0.3,
    "efficientnet_v2_s": 8.4,
    "efficientnet_v2_m": 24.0,
}


def build_model(arch: str = "alexnet", num_classes: int = 38, freeze_backbone: bool = True) -> nn.Module:
    arch = arch.lower()

    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unknown arch '{arch}'. Choose one of {SUPPORTED_ARCHS}.")

    input_size = INPUT_SIZES[arch]
    print(f"[model] building '{arch}' (input {input_size}x{input_size}, "
          f"~{_APPROX_GFLOPS[arch]} GFLOPs/image) ...", flush=True)
    print(f"[model] fetching pretrained ImageNet weights for '{arch}' "
          f"(first time for this arch = a real download from download.pytorch.org; "
          f"already-cached archs load instantly from ~/.cache/torch/hub) ...",
          flush=True)
    t0 = time.time()

    try:
        if arch == "alexnet":
            model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
            if freeze_backbone:
                for param in model.features.parameters():
                    param.requires_grad = False
            in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_features, num_classes)

        elif arch == "vgg16":
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            if freeze_backbone:
                for param in model.features.parameters():
                    param.requires_grad = False
            in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_features, num_classes)

        elif arch == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            if freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

        elif arch == "mobilenet":
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            if freeze_backbone:
                for param in model.features.parameters():
                    param.requires_grad = False
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)

        elif arch == "efficientnet_v2_s":
            model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            if freeze_backbone:
                for param in model.features.parameters():
                    param.requires_grad = False
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)

        elif arch == "efficientnet_v2_m":
            model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
            if freeze_backbone:
                for param in model.features.parameters():
                    param.requires_grad = False
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)

    except Exception as e:
        # DO NOT hide this -- print exactly what failed and where, per Task 5.
        print(f"[model] FAILED to build '{arch}' after {time.time() - t0:.1f}s: "
              f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise

    elapsed = time.time() - t0
    trainable = count_trainable_params(model)
    print(f"[model] '{arch}' ready in {elapsed:.1f}s "
          f"({'weights downloaded' if elapsed > 3 else 'weights loaded from cache'}) "
          f"| trainable params: {trainable:,}", flush=True)

    return model


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


if __name__ == "__main__":
    for arch_name in SUPPORTED_ARCHS:
        input_size = INPUT_SIZES[arch_name]
        model = build_model(arch=arch_name, num_classes=38, freeze_backbone=True)
        dummy_input = torch.randn(2, 3, input_size, input_size)
        output = model(dummy_input)
        print(f"[{arch_name}] output shape: {tuple(output.shape)} "
              f"| trainable params: {count_trainable_params(model):,}")
