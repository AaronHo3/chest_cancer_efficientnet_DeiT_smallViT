"""
Evaluate trained model on the test set.
Loads best checkpoint, runs inference, prints classification report and confusion matrix.
"""
import argparse
import os
import sys

import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import config
from data.dataset import CLASS_NAMES, get_dataloaders
from models import get_device, get_model


def main():
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

    device = get_device(prefer_mps=not args.no_mps)
    print(f"Device: {device}, Checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt.get("model_name", args.model)
    num_classes = ckpt.get("num_classes", config.NUM_CLASSES)

    model = get_model(model_name, num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, _, test_loader = get_dataloaders(
        args.data_root,
        image_size=config.IMG_SIZE,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
    )

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test"):
            images = images.to(device)
            logits = model(images)
            pred = logits.argmax(dim=1).cpu()
            all_preds.extend(pred.tolist())
            all_labels.extend(labels.tolist())

    labels_names = CLASS_NAMES[:num_classes]
    print("\n--- Classification report ---")
    print(classification_report(all_labels, all_preds, target_names=labels_names, digits=4))

    print("--- Confusion matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print("(rows=true, cols=pred)")

    # Optional: save confusion matrix plot
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(labels_names, rotation=45, ha="right")
        ax.set_yticklabels(labels_names)
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion matrix (test set)")
        plt.tight_layout()
        out_path = os.path.join(config.CHECKPOINT_DIR, f"confusion_matrix_{model_name}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"\nConfusion matrix plot saved to {out_path}")
    except Exception as e:
        print(f"(Could not save plot: {e})")


if __name__ == "__main__":
    main()
