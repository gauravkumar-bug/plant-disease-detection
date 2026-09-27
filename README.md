# Harnessing Deep Learning for Early Plant Disease Detection

Transfer-learning pipeline (AlexNet / VGG16 / ResNet50 / MobileNetV2) for
classifying plant leaf diseases using the Plant Village dataset, matching
the project's stated methodology: dataset pre-processing → pretrained CNN +
transfer learning → evaluation → model comparison → inference.

Note: the full Plant Village dataset has 87,000 images / 38 classes across
14 species. Some Kaggle mirrors (e.g. the "PlantVillage" folder version) only
include Pepper, Potato, and Tomato — 15 classes. Pass `--num_classes` to match
whichever version you actually downloaded.

## 1. Project structure

```
plant_disease_project/
├── requirements.txt
├── README.md
├── data/                    # dataset goes here (train/ and val/ folders)
├── outputs/                 # checkpoints, history, plots get saved here
└── src/
    ├── model.py             # builds AlexNet / VGG16 / ResNet50 / MobileNetV2
    ├── dataset.py            # dataloaders + preprocessing/augmentation
    ├── split_dataset.py      # splits raw Kaggle download into train/val (80/20)
    ├── train.py              # training loop with checkpointing
    ├── evaluate.py           # accuracy, precision/recall/F1, confusion matrix
    ├── predict.py             # single-image inference
    ├── plot_history.py        # training curve plots for one model
    └── compare_models.py      # trains all 4 models, saves bar-chart comparison
```

## 2. Setup

```bash
pip install -r requirements.txt
```

## 3. Get the dataset

Download the Plant Village dataset from Kaggle:
https://www.kaggle.com/datasets/emmarex/plantdisease

Then split it into `data/train/` and `data/val/` (80/20, matching the report):

```bash
python src/split_dataset.py --source /path/to/raw/PlantVillage --dest ./data --val_ratio 0.2
```

## 4. Train a single model

```bash
# AlexNet (matches the original project report, 96.63% accuracy achieved there)
python src/train.py --data_dir ./data --arch alexnet --epochs 15 --batch_size 32 --num_classes 15

# VGG16 / ResNet50 / MobileNetV2 (swap --arch)
python src/train.py --data_dir ./data --arch resnet50 --epochs 15 --batch_size 16 --num_classes 15
```

By default only the final classifier layer is trained (`freeze_backbone=True`)
— the pretrained convolutional weights are reused as-is, matching the
"transfer learning" approach described in the report. This trains much faster
and needs less data than fine-tuning the full network.

Best checkpoint is saved to `outputs/best_<arch>.pth`, along with
`class_names.json` and `history_<arch>.json`.

## 5. Evaluate

```bash
python src/evaluate.py --checkpoint ../outputs/best_alexnet.pth --data_dir ./data
```

Prints per-class precision/recall/F1 (not just overall accuracy) and saves
a confusion matrix heatmap — useful for spotting which specific diseases
get confused with each other.

## 6. Plot training curves

```bash
python src/plot_history.py --history ../outputs/history_alexnet.json
```

## 7. Predict on a new image

```bash
python src/predict.py --checkpoint ../outputs/best_alexnet.pth --image ./sample_leaf.jpg
```

Prints the top-3 predicted disease classes with confidence scores — this is
the piece that would sit behind a real farmer-facing app.

## 8. Compare all 4 models (bar chart)

Trains AlexNet, VGG16, ResNet50, and MobileNetV2 back-to-back on the same
data and produces one combined bar-chart image comparing accuracy, loss,
training time, and parameter count — the exact chart to show in your report
or interview for "why did you choose this model?":

```bash
python src/compare_models.py --data_dir ./data --epochs 5 --num_classes 15
```

- `--epochs` is small by default (5) since this is a *comparison* run, not
  final training — once you know the winner, train it properly with `train.py`
  using more epochs.
- Output: `outputs/model_comparison.png` (4-panel bar chart) and
  `outputs/comparison_results.json` (raw numbers).
- Restrict to specific models with `--archs alexnet vgg16` if you don't want
  all four.

## Notes on the four architectures

| Model | Depth/Design | Params (full) | Strength | Trade-off |
|---|---|---|---|---|
| **AlexNet** | 8 layers, large early filters | ~60M | Simple, fast baseline | Lower accuracy ceiling |
| **VGG16** | 16 layers, uniform 3×3 filters | ~138M | Strong accuracy | Heaviest, slow, overfit-prone |
| **ResNet50** | 50 layers, residual/skip connections | ~25M | Very deep without vanishing gradients, strong accuracy | More complex architecture |
| **MobileNetV2** | Depthwise-separable convs | ~3.5M | Tiny, fastest, mobile/edge-friendly | Usually lowest accuracy of the four |

Run `compare_models.py` to get your actual numbers rather than relying on
these general trends — dataset size, class count, and epochs all affect
which one actually wins for your specific data.

## Web frontend (upload photo or use camera, browser UI)

See `webapp/README.md` for a simple local web app: a FastAPI backend that
serves your trained model, plus a single-page HTML frontend with photo
upload and live camera capture, no terminal commands needed once it's
running. Works with MPS (Apple GPU) on Mac M1/M2/M3 automatically.

## Environment note

Model weights are downloaded from `download.pytorch.org` the first time each
architecture is built. This requires normal internet access (it works on a
regular machine/Colab; it will not work in network-restricted sandboxes).
