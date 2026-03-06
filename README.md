# Chest Cancer CT Classification

Classify chest CT scans into **four classes**: three NSCLC types (**Adenocarcinoma**, **Large cell carcinoma**, **Squamous cell carcinoma**) plus **Normal**, using the [Chest CT-Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) dataset.

**NOTICE:** This project is meant purely for learning and discovery.

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
- **DeiT-Tiny** — Small ViT (~5.7M params), (CPU or MPS); use `MODEL=deit_tiny`.
- **ViT-Small** — Slightly larger ViT (~22M params); use `MODEL=vit_small`.

Select via `config.MODEL` or env `MODEL` (e.g. `MODEL=deit_tiny` or `MODEL=efficientnet_b2`). All models use **4 classes** (3 cancer types + normal). On Apple Silicon, PyTorch will use MPS when available. Details and device notes are in **DESIGN.md**.

## Results 

Metrics below are from `evaluate.py` on the held-out test set (saved to `checkpoints/*_metrics.json`).

### Summary

| Model              | Accuracy | Balanced accuracy | Params   |
|--------------------|----------|-------------------|----------|
| **DeiT-Tiny**      | 81.27%   | 82.70%            | ~5.7M    |
| **EfficientNet-B2**| **90.79%** | **91.74%**      | ~9M      |
| **ViT-Small**      | 88.57%   | 89.94%            | ~22M     |

EfficientNet-B2 achieves the best test accuracy and balanced accuracy.

### Classification report

Per-class precision, recall, and F1-score on the test set.

#### DeiT-Tiny

| Class                  | Precision | Recall | F1-score | Support |
|------------------------|-----------|--------|----------|---------|
| adenocarcinoma         | 0.75      | 0.88   | 0.81     | 120     |
| large_cell_carcinoma   | 0.67      | 0.84   | 0.75     | 51      |
| squamous_cell_carcinoma| 0.95      | 0.60   | 0.73     | 90      |
| normal                 | 1.00      | 0.98   | 0.99     | 54      |
| **accuracy**           |           |        | **0.81** | 315     |
| **macro avg**          | 0.84      | 0.83   | 0.82     |         |
| **weighted avg**       | 0.84      | 0.81   | 0.81     |         |

#### EfficientNet-B2

| Class                  | Precision | Recall | F1-score | Support |
|------------------------|-----------|--------|----------|---------|
| adenocarcinoma         | 0.91      | 0.88   | 0.89     | 120     |
| large_cell_carcinoma   | 0.90      | 0.90   | 0.90     | 51      |
| squamous_cell_carcinoma| 0.86      | 0.91   | 0.89     | 90      |
| normal                 | 1.00      | 0.98   | 0.99     | 54      |
| **accuracy**           |           |        | **0.91** | 315     |
| **macro avg**          | 0.92      | 0.92   | 0.92     |         |
| **weighted avg**       | 0.91      | 0.91   | 0.91     |         |

#### ViT-Small

| Class                  | Precision | Recall | F1-score | Support |
|------------------------|-----------|--------|----------|---------|
| adenocarcinoma         | 0.85      | 0.91   | 0.88     | 120     |
| large_cell_carcinoma   | 0.79      | 0.94   | 0.86     | 51      |
| squamous_cell_carcinoma| 0.95      | 0.77   | 0.85     | 90      |
| normal                 | 1.00      | 0.98   | 0.99     | 54      |
| **accuracy**           |           |        | **0.89** | 315     |
| **macro avg**          | 0.90      | 0.90   | 0.89     |         |
| **weighted avg**       | 0.89      | 0.89   | 0.89     |         |

### Figures

Evaluation figures are saved in `checkpoints/` for each model:

- `*_confusion_matrix.png` — confusion matrix (counts and proportions)
- `*_roc_curves.png` — ROC curves (one-vs-rest) with AUC
- `*_per_class_metrics.png` — bar chart of precision, recall, F1 per class
- `*_evaluation_dashboard.png` — combined summary figure
- `*_training_curves.png` — loss and accuracy vs epoch
