"""
Train 4-class chest cancer CT model (EfficientNet-B2 or DeiT-Tiny).
Uses config for paths and hyperparameters; saves best checkpoint by validation accuracy.
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
from tqdm import tqdm

import config
from data.dataset import get_dataloaders
from models import get_device, get_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one full pass over the training set; update weights and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # clear gradients from previous batch
        logits = model(images)
        loss = criterion(logits, labels)  # cross-entropy
        loss.backward()  # compute gradients
        optimizer.step()  # update weights
        total_loss += loss.item()
        pred = logits.argmax(dim=1)  # predicted class (index with highest logit)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total if total else 0.0


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Run model on loader without updating weights; return average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader) if len(loader) else 0.0, correct / total if total else 0.0


def main():
    # --- Parse arguments and check data path ---
    parser = argparse.ArgumentParser(description="Train chest cancer CT classifier")
    parser.add_argument("--model", type=str, default=config.MODEL, choices=["efficientnet_b2", "deit_tiny", "vit_small"])
    parser.add_argument("--data_root", type=str, default=config.DATA_ROOT, help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE, help="Early stop after N epochs without improvement")
    parser.add_argument("--checkpoint_dir", type=str, default=config.CHECKPOINT_DIR)
    parser.add_argument("--no_mps", action="store_true", help="Disable MPS (force CPU on Mac)")
    args = parser.parse_args()

    if not args.data_root or not os.path.isdir(os.path.join(args.data_root, "Data")):
        print("ERROR: DATA_ROOT not set or Data/ not found. Run: python -m data.download")
        sys.exit(1)

    device = get_device(prefer_mps=not args.no_mps)
    print(f"Device: {device}")

    # --- Data loaders: train (with augmentation), valid (no augmentation), test not used during training ---
    train_loader, valid_loader, _ = get_dataloaders(
        args.data_root,
        image_size=config.IMG_SIZE,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
    )
    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

    # --- Model, loss, optimizer, and learning-rate scheduler ---
    model = get_model(args.model, num_classes=config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)  # reduce lr when val acc plateaus

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0  # epochs since last improvement (for early stopping)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # --- Training loop: one epoch at a time, validate after each, save best model, optional early stop ---
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
        scheduler.step(val_acc)  # may reduce learning rate if val acc didn't improve
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch}: train loss={train_loss:.4f} acc={train_acc:.4f} | val loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            path = os.path.join(args.checkpoint_dir, f"best_{args.model}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "model_name": args.model,
                "num_classes": config.NUM_CLASSES,
            }, path)
            print(f"  -> Saved best to {path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, val_acc={best_val_acc:.4f})")
                break

    print(f"Done. Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")

    # --- Save training curves (loss and accuracy vs epoch) as a figure ---
    try:
        import matplotlib.pyplot as plt
        epochs_range = range(1, len(history["train_loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs_range, history["train_loss"], "b-", label="Train")
        ax1.plot(epochs_range, history["val_loss"], "r-", label="Valid")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss vs Epoch")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(epochs_range, history["train_acc"], "b-", label="Train")
        ax2.plot(epochs_range, history["val_acc"], "r-", label="Valid")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy vs Epoch")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(args.checkpoint_dir, f"{args.model}_training_curves.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Training curves saved to {plot_path}")
    except Exception as e:
        print(f"Could not save training curves: {e}")


if __name__ == "__main__":
    main()
