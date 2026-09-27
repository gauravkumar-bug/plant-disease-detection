"""
app.py
------
FastAPI backend for the plant disease detection frontend. Loads a trained
checkpoint once at startup and exposes a /predict endpoint that accepts an
uploaded image and returns the top-k predicted disease classes.

Usage:
    cd webapp/backend
    python3 app.py --checkpoint ../../outputs/compare_alexnet.pth

Then open webapp/frontend/index.html in your browser (or serve it -- see
the frontend/README section) and it will call this API at
http://localhost:8000.

Works on Mac M1/M2/M3: automatically uses the Apple GPU (MPS) if available,
same device-selection logic as the rest of the project (model.get_device()).
"""

import argparse
import io
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Make src/ importable (model.py, dataset.py live there)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from model import build_model, get_device  # noqa: E402
from dataset import get_transforms  # noqa: E402

app = FastAPI(title="Plant Disease Detection API")

# Allow the frontend (opened as a local file or served from any origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Populated at startup by load_model()
STATE = {"model": None, "class_names": None, "arch": None, "device": None, "val_transform": None}


def load_model(checkpoint_path: str):
    device = get_device()
    print(f"Loading checkpoint from {checkpoint_path} onto device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    arch = checkpoint["arch"]
    class_names = checkpoint["class_names"]

    model = build_model(arch=arch, num_classes=checkpoint["num_classes"], freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _, val_transform = get_transforms(arch)

    STATE["model"] = model
    STATE["class_names"] = class_names
    STATE["arch"] = arch
    STATE["device"] = device
    STATE["val_transform"] = val_transform

    print(f"Loaded {arch} with {len(class_names)} classes. Ready.")


@app.get("/health")
def health():
    return {
        "status": "ok" if STATE["model"] is not None else "model not loaded",
        "arch": STATE["arch"],
        "device": str(STATE["device"]),
        "num_classes": len(STATE["class_names"]) if STATE["class_names"] else 0,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        raw_bytes = await file.read()
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    input_tensor = STATE["val_transform"](image).unsqueeze(0).to(STATE["device"])

    with torch.no_grad():
        outputs = STATE["model"](input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        top_k = min(top_k, len(STATE["class_names"]))
        top_probs, top_indices = torch.topk(probabilities, top_k)

    predictions = [
        {
            "class_name": STATE["class_names"][idx],
            "confidence": round(prob.item() * 100, 2),
        }
        for prob, idx in zip(top_probs, top_indices)
    ]

    return {"predictions": predictions, "arch": STATE["arch"]}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the plant disease detection API.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained .pth checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_model(args.checkpoint)
    uvicorn.run(app, host=args.host, port=args.port)
