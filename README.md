# Chest Cancer CT Classification

Classify chest CT scans into **four classes**: three NSCLC types (**Adenocarcinoma**, **Large cell carcinoma**, **Squamous cell carcinoma**) plus **Normal**, using the [Chest CT-Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) dataset.

## Setup

```bash
pip install -r requirements.txt
```

Ensure you have Kaggle API credentials set up if needed for `kagglehub` (see [kagglehub docs](https://github.com/Kaggle/kagglehub)).

## Quick start

1. **Download dataset**
   ```bash
   python -m data.download
   ```
   Set `DATASET_PATH` to the printed path, or pass it to scripts.

2. **Inspect data**
   ```bash
   python scripts/inspect_data.py
   # or with path:
   python scripts/inspect_data.py /path/to/downloaded/dataset
   ```

3. **Design and implementation plan**  
   See [DESIGN.md](DESIGN.md) for architecture choices, data pipeline, training strategy, and suggested project layout.

## Models

- **EfficientNet-B2** — Strong baseline (torchvision), 224×224.
- **DeiT-Tiny** — Small ViT (~5.7M params), **MacBook-friendly** (CPU or MPS); use `MODEL=deit_tiny`.
- **ViT-Small** — Slightly larger ViT (~22M params); use `MODEL=vit_small`.

Select via `config.MODEL` or env `MODEL` (e.g. `MODEL=deit_tiny` or `MODEL=efficientnet_b2`). All models use **4 classes** (3 cancer types + normal). On Apple Silicon, PyTorch will use MPS when available. Details and device notes are in **DESIGN.md**.
