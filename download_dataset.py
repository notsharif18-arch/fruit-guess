import os
import zipfile
from pathlib import Path

import kagglehub


def download_dataset():
    dataset_dir = Path("dataset")

    if dataset_dir.exists():
        print("✅ Dataset already exists.")
        return

    print("⬇ Downloading Fruits-360 dataset...")

    path = kagglehub.dataset_download("moltean/fruits")

    print(f"Downloaded to: {path}")

    dataset_dir.mkdir(exist_ok=True)

    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = dataset_dir / item

        if os.path.isdir(src):
            os.rename(src, dst)
        elif src.endswith(".zip"):
            with zipfile.ZipFile(src, "r") as zip_ref:
                zip_ref.extractall(dataset_dir)

    print("✅ Dataset ready!")


if __name__ == "__main__":
    download_dataset()
