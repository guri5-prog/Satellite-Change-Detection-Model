import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ChangeDataset(Dataset):
    """
    Loads change detection datasets from:
    data/standardized/{split}/{dataset_name}/{A,B,mask}
    - A: before image
    - B: after image
    - mask: binary change mask (PNG/TIFF)
    """
    def _init_(self, root_dir, split='train', image_size=96, use_augmentations=True):
        self.samples = []
        split_dir = os.path.join(root_dir, split)

        if not os.path.exists(split_dir):
            raise RuntimeError(f"Split folder not found: {split_dir}")

        print(f"[*] Searching for datasets in '{split_dir}'...")
        datasets_found = 0

        for dataset_name in sorted(os.listdir(split_dir)):
            dataset_path = os.path.join(split_dir, dataset_name)
            if not os.path.isdir(dataset_path):
                continue

            A_path = os.path.join(dataset_path, "A")
            B_path = os.path.join(dataset_path, "B")
            mask_path = os.path.join(dataset_path, "mask")

            # Ensure all required folders exist
            if not (os.path.isdir(A_path) and os.path.isdir(B_path) and os.path.isdir(mask_path)):
                print(f"  ⚠ Skipping '{dataset_name}': Missing 'A', 'B', or 'mask' folder")
                continue

            # Collect file lists
            A_files = sorted([f for f in os.listdir(A_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            B_files = sorted([f for f in os.listdir(B_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            M_files = sorted([f for f in os.listdir(mask_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

            if not A_files or not B_files or not M_files:
                print(f"  ⚠ Skipping '{dataset_name}': Empty folders")
                continue

            if len(A_files) != len(B_files) or len(A_files) != len(M_files):
                print(f"  ⚠ Skipping '{dataset_name}': Mismatched file counts")
                continue

            datasets_found += 1
            print(f"  ✅ Found dataset: {dataset_name}")

            for a_file, b_file, m_file in zip(A_files, B_files, M_files):
                self.samples.append({
                    "a_path": os.path.join(A_path, a_file),
                    "b_path": os.path.join(B_path, b_file),
                    "mask_path": os.path.join(mask_path, m_file)
                })

        if datasets_found == 0:
            raise RuntimeError(f"❌ No valid datasets found in '{split_dir}'.")

        print(f"[✔] Total valid image triplets: {len(self.samples)}")

        # Albumentations transforms
        additional_targets = {'image0': 'image'}
        if use_augmentations and split == 'train':
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=(-15, 15), p=0.5),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ], additional_targets=additional_targets)
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ], additional_targets=additional_targets)

    def _len_(self):
        return len(self.samples)

    def _getitem_(self, idx):
        sample = self.samples[idx]
        a_img = cv2.cvtColor(cv2.imread(sample["a_path"]), cv2.COLOR_BGR2RGB)
        b_img = cv2.cvtColor(cv2.imread(sample["b_path"]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)

        # Convert mask to binary (0 or 1)
        mask = (mask > 127).astype('float32')

        transformed = self.transform(image=a_img, image0=b_img, mask=mask)
        a_tensor = transformed['image']
        b_tensor = transformed['image0']
        mask_tensor = transformed['mask'].unsqueeze(0)  # add channel dim

        return {'a': a_tensor, 'b': b_tensor, 'mask': mask_tensor}