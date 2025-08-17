import os
import shutil
import random
from pathlib import Path

# Adjust these
STANDARDIZED_ROOT = Path("data/standardized")
DATASETS = {
    "dfc2020": ["train", "val", "test"],
    "LEVIR-CD+": ["train", "test"],
    "dsifn": ["train", "test"],
    "kaggle": [],  # no splits, all data in one folder
    "levir": [],
    "oscd": []
}

SPLITS = ["train", "val", "test"]
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

random.seed(SEED)

def move_existing_splits(dataset_name, splits):
    for split in splits:
        src = STANDARDIZED_ROOT / dataset_name / split
        if not src.exists():
            print(f"[!] Missing expected folder: {src}")
            continue
        dest = STANDARDIZED_ROOT / split / dataset_name
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Moving {src} → {dest}")
        for item in src.iterdir():
            shutil.move(str(item), str(dest / item.name))
        # Remove empty folder
        src.rmdir()
    # Remove original dataset folder if empty
    orig_folder = STANDARDIZED_ROOT / dataset_name
    if orig_folder.exists() and not any(orig_folder.iterdir()):
        orig_folder.rmdir()

def split_and_move(dataset_name):
    # No splits, so all data is in STANDARDIZED_ROOT/dataset_name/{A,B,label,mask}
    src = STANDARDIZED_ROOT / dataset_name
    if not src.exists():
        print(f"[!] Dataset folder missing: {src}")
        return
    all_files = {}
    # Collect file lists for each category (A,B,label,mask)
    for subfolder in ["A", "B", "label", "mask"]:
        folder = src / subfolder
        if not folder.exists():
            print(f"[!] Missing subfolder {subfolder} in {src}")
            return
        all_files[subfolder] = list(folder.iterdir())
        all_files[subfolder].sort()  # ensure same order for matching files

    # Number of samples
    num_samples = len(all_files["A"])
    assert all(len(files) == num_samples for files in all_files.values()), "Mismatch in number of files per subfolder!"

    indices = list(range(num_samples))
    random.shuffle(indices)

    num_val = int(num_samples * VAL_RATIO)
    num_test = int(num_samples * TEST_RATIO)
    num_train = num_samples - num_val - num_test

    splits_indices = {
        "train": indices[:num_train],
        "val": indices[num_train:num_train+num_val],
        "test": indices[num_train+num_val:]
    }

    for split in SPLITS:
        for subfolder in ["A", "B", "label", "mask"]:
            dest_folder = STANDARDIZED_ROOT / split / dataset_name / subfolder
            dest_folder.mkdir(parents=True, exist_ok=True)

    # Move files to respective folders
    for subfolder in ["A", "B", "label", "mask"]:
        files = all_files[subfolder]
        for split in SPLITS:
            for idx in splits_indices[split]:
                src_file = files[idx]
                dest_file = STANDARDIZED_ROOT / split / dataset_name / subfolder / src_file.name
                shutil.move(str(src_file), str(dest_file))

    # Remove original dataset folder if empty
    if src.exists() and not any(src.iterdir()):
        src.rmdir()

def main():
    for dataset_name, splits in DATASETS.items():
        if splits:
            print(f"Processing dataset with splits: {dataset_name}")
            move_existing_splits(dataset_name, splits)
        else:
            print(f"Processing dataset with no splits: {dataset_name}")
            split_and_move(dataset_name)

if __name__ == "__main__":
    main()



