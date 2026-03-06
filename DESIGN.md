# Chest Cancer CT Classification — Design & Plan

Classify chest CT scans into **four classes**: three NSCLC types (Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma) plus **Normal**, using the [Chest CT-Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) dataset (via `kagglehub`).

---

## 1. Dataset Overview

- **Source**: `mohamedhanyyy/chest-ctscan-images` (Kaggle).
- **Splits**: `Data/Train` (70%), `Data/Test` (20%), `Data/Valid` (10%).
- **Classes** (4-way classification):
  - **Adenocarcinoma** — outer lung, ~40% of NSCLC.
  - **Large cell carcinoma** — 10–15% NSCLC, fast-growing.
  - **Squamous cell carcinoma** — central airways, ~30% NSCLC.
  - **Normal** — no cancer; healthy lung CTs.

---

## 2. Recommended Approach

### 2.1 Transfer learning 

- **Preferred backbone**: **EfficientNet-B2 or EfficientNet-B3** (good accuracy/speed trade-off; literature shows strong results on lung/CT tasks).
- **Alternatives**: ResNet-50 (simpler, faster), EfficientNetV2-S (if you want to try a newer variant).
- **Framework**: PyTorch + `torchvision` (pretrained on ImageNet); fine-tune the classifier head and optionally last 1–2 stages of the backbone.

### 2.2 Why this works well

- Limited medical data → transfer learning is standard and effective.
- CT slices are similar enough to natural images that ImageNet features help.
- EfficientNet gives strong performance with reasonable compute; ResNet-50 is a solid baseline.

### 2.3 Small Vision Transformer (ViT) — MacBook-friendly

- **DeiT-Tiny** (`deit_tiny_patch16_224`): ~5.7M params, 224×224 input. Runs well on **MacBook (CPU or MPS)** and is included as a second model option.
- **ViT-Small** (`vit_small_patch16_224`): ~22M params; use if you have more RAM/time.
- **Framework**: Use **timm** (`timm.create_model(..., pretrained=True)`) and replace the classifier head with a 4-class linear layer. PyTorch **MPS** (Apple Silicon) is supported; fallback to CPU if needed.
- Same data pipeline (224×224, ImageNet norm) works for both CNN and ViT.

---

## 3. Data Pipeline

### 3.1 Preprocessing

- **Load**: Read images from `Train`, `Test`, `Valid` (e.g. PIL/OpenCV → tensors).
- **Resize**: Fixed size (e.g. 224×224 for EfficientNet/ResNet, or 384 if using a larger model).
- **Normalize**: Use ImageNet mean/std (e.g. `[0.485, 0.456, 0.406]`, `[0.229, 0.224, 0.225]`) to match pretrained weights.

### 3.2 Data augmentation (training only)

- **Safe**: Random horizontal flip, small rotation (±10–15°), slight zoom/crop.
- **Moderate**: Color jitter (brightness/contrast) with small factors so as not to distort CT appearance.
- **Optional**: Random erasing / Cutout to improve robustness.
- **Avoid**: Aggressive geometric or intensity transforms that would not occur in real CT (e.g. heavy distortion, extreme contrast).

### 3.3 Class balance

- Check class counts in `Train` (and Valid). If imbalanced:
  - **Weighted cross-entropy** (inverse frequency or sqrt-inverse).
  - **Oversampling** (e.g. repeat minority classes or use a weighted sampler).
  - Prefer weighted loss + optional oversampling for stability.

---

## 4. Model Architecture 

### 4.1 Baseline: EfficientNet-B2

- **Backbone**: `torchvision.models.efficientnet_b2(weights=...)`.
- **Classifier**: Replace `classifier` with `nn.Sequential(..., Linear(256 → 4))`.
- **Output**: 4 logits (Adenocarcinoma, Large cell, Squamous, Normal).

### 4.2 Small ViT (DeiT-Tiny)

- **Backbone**: `timm.create_model("deit_tiny_patch16_224", pretrained=True)`.
- **Head**: Replace `model.head` with `nn.Linear(feat_dim, 4)`. Get `feat_dim` from `model.num_features` (192 for DeiT-Tiny).
- **Input**: 224×224; same ImageNet normalization as EfficientNet.
- **Device**: Prefer MPS on Apple Silicon, else CPU (or CUDA if running elsewhere). Batch size 16–24 is often fine on a MacBook.

### 4.3 Training strategy

- **Optimizer**: Adam or AdamW (e.g. `lr=1e-4`, weight decay `1e-2`).
- **Scheduler**: Cosine annealing or ReduceLROnPlateau (e.g. on validation loss).
- **Epochs**: 20–40; use validation loss for early stopping.
- **Freezing**: Optionally freeze backbone for 1–3 epochs, then unfreeze and fine-tune with a smaller lr (e.g. 2× smaller).

---

## 5. Evaluation

- **Metrics**: Accuracy, **balanced accuracy**, **per-class precision/recall/F1**, **confusion matrix**, **ROC-AUC (one-vs-rest)**.
- **Reports**: Classification report (sklearn) + confusion matrix plot; save best model by validation balanced accuracy or AUC.
- **Split usage**: Train on `Train`, tune/early-stop on `Valid`, report final metrics on `Test` (single final evaluation to avoid overfitting to test).

---

## 6. Project Structure 

```
chest_cancer_recognition/
├── DESIGN.md              # This file
├── README.md
├── requirements.txt
├── config.py               # Paths, image size, batch size, lr, etc.
├── data/
│   ├── dataset.py          # PyTorch Dataset (load by class folders)
│   └── download.py         # kagglehub download + optional inspect
├── models/
│   ├── __init__.py         # get_model(name, num_classes), get_device()
│   ├── efficient_net.py    # Wrapper: EfficientNet-B2 + custom head
│   └── vit_small.py        # Wrapper: DeiT-Tiny (timm) + 4-class head
├── train.py                # Training loop, checkpointing, logging
├── evaluate.py             # Load best model, run on Test, print metrics
└── scripts/
    └── inspect_data.py    # Dataset stats, class counts, sample images
```

---

## 7. Implementation Order

1. **Download & inspect** — Run `download.py`, then `inspect_data.py` to confirm folder layout and class balance.
2. **Data module** — Implement `dataset.py` (Dataset + DataLoaders with augmentations and ImageNet normalization).
3. **Model** — Implement EfficientNet-B2 + 4-class head in `models/efficient_net.py`.
4. **Training** — Implement `train.py` (loss: cross-entropy with optional class weights; validation on `Valid`).
5. **Evaluation** — Implement `evaluate.py` on `Test`; log metrics and confusion matrix.
6. **Small ViT** — Add DeiT-Tiny in `models/vit_small.py`; select via `--model deit_tiny` or `--model efficientnet_b2`. Use same train/eval pipeline; device auto-selects MPS (MacBook) or CPU/CUDA.

---

## 8. Quick Reference: Class Names

| Index | Class name              |
|-------|-------------------------|
| 0     | Adenocarcinoma          |
| 1     | Large cell carcinoma    |
| 2     | Squamous cell carcinoma |
| 3     | Normal                  |

Use consistent string labels (e.g. folder names) and map to these indices in the dataset class.

---

## 9. Risks & Mitigations

- **Class imbalance** → Weighted loss + oversampling/sampler.
- **Overfitting** → Dropout, augmentation, early stopping, optional L2.
- **Small dataset** → Rely on transfer learning and moderate augmentation; avoid very large models (e.g. EfficientNet-B7) unless you add more data.
