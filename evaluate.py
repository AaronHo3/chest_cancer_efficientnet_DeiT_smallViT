"""
Evaluate trained model on the test set.
Loads best checkpoint, runs inference, prints classification report and confusion matrix.
Generates figures and saves metrics to checkpoints/ for later use.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_fscore_support,
)
from tqdm import tqdm

import config
from data.dataset import CLASS_NAMES, get_dataloaders
from models import get_device, get_model


def format_classification_report_table(labels_names, precision, recall, f1, support, accuracy):
    """Build classification report as a formatted table string (for console and .txt file)."""
    n = len(labels_names)
    # Column widths: class name column fits longest label; number columns fixed width
    w_name = max(len(s) for s in labels_names) + 1
    w_num = 10
    head = f"{'Class':<{w_name}} {'Precision':>{w_num}} {'Recall':>{w_num}} {'F1-score':>{w_num}} {'Support':>{w_num}}"
    sep = "-" * len(head)
    lines = [head, sep]
    # One row per class
    for i in range(n):
        lines.append(
            f"{labels_names[i]:<{w_name}} {precision[i]:>{w_num}.4f} {recall[i]:>{w_num}.4f} {f1[i]:>{w_num}.4f} {int(support[i]):>{w_num}}"
        )
    lines.append(sep)
    # Summary rows: macro = simple average across classes; weighted = by support count
    macro_p, macro_r, macro_f = precision.mean(), recall.mean(), f1.mean()
    total_support = support.sum()
    weighted_p = (precision * support).sum() / total_support
    weighted_r = (recall * support).sum() / total_support
    weighted_f = (f1 * support).sum() / total_support
    lines.append(f"{'accuracy':<{w_name}} {'':>{w_num}} {'':>{w_num}} {accuracy:>{w_num}.4f} {int(total_support):>{w_num}}")
    lines.append(f"{'macro avg':<{w_name}} {macro_p:>{w_num}.4f} {macro_r:>{w_num}.4f} {macro_f:>{w_num}.4f} {int(total_support):>{w_num}}")
    lines.append(f"{'weighted avg':<{w_name}} {weighted_p:>{w_num}.4f} {weighted_r:>{w_num}.4f} {weighted_f:>{w_num}.4f} {int(total_support):>{w_num}}")
    return "\n".join(lines)


def main():
    # --- Parse arguments: which model, where checkpoint and data live ---
    parser = argparse.ArgumentParser(description="Evaluate chest cancer CT classifier on test set")
    parser.add_argument("--model", type=str, default=config.MODEL, choices=["efficientnet_b2", "deit_tiny", "vit_small"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file (default: checkpoints/best_<model>.pt)")
    parser.add_argument("--data_root", type=str, default=config.DATA_ROOT)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--no_mps", action="store_true", help="Disable MPS (force CPU)")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or os.path.join(config.CHECKPOINT_DIR, f"best_{args.model}.pt")
    if not os.path.isfile(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Train first: python train.py --model", args.model)
        sys.exit(1)

    # --- Load model from checkpoint (no need for pretrained weights; we use trained ones) ---
    device = get_device(prefer_mps=not args.no_mps)
    print(f"Device: {device}, Checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt.get("model_name", args.model)
    num_classes = ckpt.get("num_classes", config.NUM_CLASSES)

    model = get_model(model_name, num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --- Load test data (same transforms as validation: resize, center crop, normalize) ---
    _, _, test_loader = get_dataloaders(
        args.data_root,
        image_size=config.IMG_SIZE,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
    )

    # --- Run model on every test batch; collect predictions, labels, and class probabilities ---
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test"):
            images = images.to(device)
            logits = model(images)  # raw scores per class
            probs = F.softmax(logits, dim=1).cpu().numpy()  # probabilities (for ROC curves)
            pred = logits.argmax(dim=1).cpu()  # predicted class index
            all_preds.extend(pred.tolist())
            all_labels.extend(labels.tolist())
            all_probs.append(probs)
    all_probs = np.vstack(all_probs)
    all_labels_arr = np.array(all_labels)
    all_preds_arr = np.array(all_preds)

    # --- Compute metrics: per-class precision/recall/F1/support, overall accuracy ---
    labels_names = CLASS_NAMES[:num_classes]
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels_arr, all_preds_arr, average=None, zero_division=0
    )
    accuracy = (all_preds_arr == all_labels_arr).mean()
    balanced_accuracy = float(recall.mean())  # mean of per-class recall (treats classes equally)

    report_table = format_classification_report_table(labels_names, precision, recall, f1, support, accuracy)
    print("\n--- Classification report ---")
    print(report_table)

    # --- Save report (.txt) and metrics (.json) so they can be used e.g. to update README ---
    out_dir = config.CHECKPOINT_DIR
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{model_name}_"
    with open(os.path.join(out_dir, f"{prefix}classification_report.txt"), "w") as f:
        f.write(f"Model: {model_name}\n\n")
        f.write(report_table)
        f.write("\n")
    metrics = {
        "model": model_name,
        "accuracy": round(float(accuracy), 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "per_class": {
            name: {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1_score": round(float(f), 4),
                "support": int(s),
            }
            for name, p, r, f, s in zip(labels_names, precision, recall, f1, support)
        },
        "macro_avg": {
            "precision": round(float(precision.mean()), 4),
            "recall": round(float(recall.mean()), 4),
            "f1_score": round(float(f1.mean()), 4),
        },
        "weighted_avg": {
            "precision": round(float((precision * support).sum() / support.sum()), 4),
            "recall": round(float((recall * support).sum() / support.sum()), 4),
            "f1_score": round(float((f1 * support).sum() / support.sum()), 4),
        },
    }
    with open(os.path.join(out_dir, f"{prefix}metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nReport saved to {out_dir}/{prefix}classification_report.txt")
    print(f"Metrics saved to {out_dir}/{prefix}metrics.json")

    print("\n--- Confusion matrix ---")
    cm = confusion_matrix(all_labels, all_preds)  # rows = true class, cols = predicted class
    print(cm)
    print("(rows=true, cols=pred)")

    # --- Generate and save figures ---
    try:
        import matplotlib.pyplot as plt

        out_dir = config.CHECKPOINT_DIR
        os.makedirs(out_dir, exist_ok=True)
        prefix = f"{model_name}_"

        # 1. Confusion matrix heatmap: normalized by row so each row sums to 1 (proportion of true class predicted as each class)
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(labels_names, rotation=30, ha="right")
        ax.set_yticklabels(labels_names)
        for i in range(num_classes):
            for j in range(num_classes):
                txt = f"{cm[i, j]}\n({cm_norm[i, j]:.0%})" if cm[i, j] > 0 else ""
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                       color="white" if cm_norm[i, j] > 0.5 else "black")
        plt.colorbar(im, ax=ax, label="Proportion of true class")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (Test Set)")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}confusion_matrix.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # 2. ROC curves: for each class, treat it as positive and rest as negative; plot TPR vs FPR at various thresholds
        fig, ax = plt.subplots(figsize=(7, 6))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(all_labels_arr == i, all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{labels_names[i]} (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves (One-vs-Rest)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}roc_curves.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # 3. Bar chart: precision, recall, F1 for each class side by side
        x = np.arange(num_classes)
        width = 0.25
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width, precision, width, label="Precision", color="#2ecc71")
        ax.bar(x, recall, width, label="Recall", color="#3498db")
        ax.bar(x + width, f1, width, label="F1-score", color="#9b59b6")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_names, rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Metrics (Test Set)")
        ax.legend()
        ax.set_ylim([0, 1.05])
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}per_class_metrics.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # 4. One figure combining confusion matrix, ROC curves, and per-class metrics
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

        ax1 = fig.add_subplot(gs[0, 0])
        im = ax1.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax1.set_xticks(range(num_classes))
        ax1.set_yticks(range(num_classes))
        ax1.set_xticklabels(labels_names, rotation=30, ha="right")
        ax1.set_yticklabels(labels_names)
        for i in range(num_classes):
            for j in range(num_classes):
                txt = str(cm[i, j]) if cm[i, j] > 0 else ""
                ax1.text(j, i, txt, ha="center", va="center", fontsize=10,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("True")
        ax1.set_title("Confusion Matrix")

        ax2 = fig.add_subplot(gs[0, 1])
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(all_labels_arr == i, all_probs[:, i])
            ax2.plot(fpr, tpr, lw=2, label=labels_names[i])
        ax2.plot([0, 1], [0, 1], "k--", lw=1)
        ax2.set_xlabel("FPR")
        ax2.set_ylabel("TPR")
        ax2.set_title("ROC Curves")
        ax2.legend(loc="lower right", fontsize=8)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, :])
        ax3.bar(x - width, precision, width, label="Precision", color="#2ecc71")
        ax3.bar(x, recall, width, label="Recall", color="#3498db")
        ax3.bar(x + width, f1, width, label="F1-score", color="#9b59b6")
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels_names, rotation=30, ha="right")
        ax3.set_ylabel("Score")
        ax3.set_title("Per-Class Metrics")
        ax3.legend()
        ax3.set_ylim([0, 1.05])
        ax3.grid(True, axis="y", alpha=0.3)

        fig.suptitle(f"Evaluation Report — {model_name} (Test Set)", fontsize=14, fontweight="bold")
        fig.savefig(os.path.join(out_dir, f"{prefix}evaluation_dashboard.png"), dpi=150, bbox_inches="tight")
        plt.close()

        print(f"\nFigures saved to {out_dir}:")
        print(f"  - {prefix}confusion_matrix.png")
        print(f"  - {prefix}roc_curves.png")
        print(f"  - {prefix}per_class_metrics.png")
        print(f"  - {prefix}evaluation_dashboard.png (summary)")
    except Exception as e:
        print(f"\nCould not generate figures: {e}")


if __name__ == "__main__":
    main()
