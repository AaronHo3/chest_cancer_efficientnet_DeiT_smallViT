"""
Inspect dataset structure and class balance.
Expects dataset root as first CLI arg, or runs download and uses that path.
Usage:
  python scripts/inspect_data.py
  python scripts/inspect_data.py /path/to/downloaded/dataset
"""
import os
import sys

def inspect(root: str) -> None:
    data_dir = os.path.join(root, "Data")
    if not os.path.isdir(data_dir):
        print("No 'Data' folder at", root)
        return

    # 4-way: 3 cancer types + normal (folder names may vary; we discover from disk)
    target_classes = [
        "adenocarcinoma",
        "large.cell.carcinoma",
        "squamous.cell.carcinoma",
        "normal",
    ]
    all_classes = set()

    for split in ("Train", "Test", "Valid"):
        split_path = os.path.join(data_dir, split)
        if not os.path.isdir(split_path):
            continue
        classes = sorted(
            d for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d))
        )
        all_classes.update(classes)
        print(f"\n{split}:")
        total = 0
        for c in classes:
            class_path = os.path.join(split_path, c)
            count = len([f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
            total += count
            print(f"  {c}: {count} images")
        print(f"  Total: {total}")

    print("\nAll class names found:", sorted(all_classes))
    print("\nFor 4-class (3 cancers + normal), use:", target_classes)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        root = sys.argv[1].rstrip("/")
    else:
        # Run download and use returned path
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data.download import download_dataset
        root = download_dataset()
    inspect(root)
