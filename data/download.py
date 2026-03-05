"""
Download the Chest CT-Scan Images dataset via kagglehub.
Run from project root: python -m data.download
"""
import os

def download_dataset():
    import kagglehub

    path = kagglehub.dataset_download("mohamedhanyyy/chest-ctscan-images")
    print("Path to dataset files:", path)
    return path


if __name__ == "__main__":
    path = download_dataset()
    # Quick sanity check: look for Data/Train, etc.
    data_dir = os.path.join(path, "Data")
    if os.path.isdir(data_dir):
        for split in ("Train", "Test", "Valid"):
            split_path = os.path.join(data_dir, split)
            if os.path.isdir(split_path):
                classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
                print(f"  {split}: {len(classes)} classes -> {classes}")
    else:
        print("  (Data/ not found; listing top-level:", os.listdir(path))
