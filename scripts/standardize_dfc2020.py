import os
import warnings
from tqdm import tqdm
import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

def normalize_name(filename):
    """Normalizes s1/s2/lc names to a common placeholder."""
    name, _ = os.path.splitext(filename)
    return name.replace("_s1_", "_sx_").replace("_s2_", "_sx_").replace("_lc_", "_sx_")

def get_normalized_filenames(folder):
    return {normalize_name(f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))}

def check_split(split_path):
    s1_path = os.path.join(split_path, "s1")
    s2_path = os.path.join(split_path, "s2")
    lc_path = os.path.join(split_path, "lc")

    if not all(os.path.exists(p) for p in [s1_path, s2_path, lc_path]):
        print(f"❌ Missing one of the subfolders (s1, s2, or lc) in {split_path}")
        return None, None, None, None

    s1_files_norm = get_normalized_filenames(s1_path)
    s2_files_norm = get_normalized_filenames(s2_path)
    lc_files_norm = get_normalized_filenames(lc_path)

    common_norm = s1_files_norm & s2_files_norm & lc_files_norm

    print(f"📂 {os.path.basename(split_path)} | Report")
    print(f"  - S1 files (normalized): {len(s1_files_norm)}")
    print(f"  - S2 files (normalized): {len(s2_files_norm)}")
    print(f"  - LC files (normalized): {len(lc_files_norm)}")
    print(f"  - Common triplets found: {len(common_norm)}")
    if common_norm:
        print(f"    Example matches: {list(common_norm)[:5]}")
    else:
        print("   ⚠️ No matching patterns found after normalization.")
    print("-" * 20)

    return s1_files_norm, s2_files_norm, lc_files_norm, common_norm

def read_tiff_as_image(path, size=(256, 256), rgb=True, is_label=False):
    """Reads a TIFF with rasterio, normalizes, resizes, and returns a PIL Image."""
    with rasterio.open(path) as src:
        img = src.read(out_shape=(src.count, size[1], size[0]), resampling=Resampling.bilinear)
        img = img.astype(np.float32)

        # Normalize from data range to 0–255
        if img.max() > 0:
            img = img / img.max() * 255.0

        img = img.clip(0, 255).astype(np.uint8)

        if is_label:
            # For label, use the first band only
            return Image.fromarray(img[0], mode="L")
        else:
            if rgb:
                if img.shape[0] >= 3:
                    img = img[:3]  # Take first 3 bands
                elif img.shape[0] == 1:
                    img = np.repeat(img, 3, axis=0)
                img = np.transpose(img, (1, 2, 0))  # (H, W, C)
                return Image.fromarray(img, mode="RGB")
            else:
                return Image.fromarray(img[0], mode="L")

def standardize_dfc2020(original_root, output_root, image_size=(256, 256)):
    splits = ['train', 'val', 'test']

    for split in splits:
        print(f"\n🔄 Processing split: {split}")
        split_input_dir = os.path.join(original_root, split)
        _, _, _, common_normalized_names = check_split(split_input_dir)

        if not common_normalized_names:
            print(f"[!] No common images found in {split}. Skipping.")
            continue

        s1_dir = os.path.join(split_input_dir, 's1')
        s2_dir = os.path.join(split_input_dir, 's2')
        label_dir = os.path.join(split_input_dir, 'lc')

        a_out = os.path.join(output_root, split, 'A')
        b_out = os.path.join(output_root, split, 'B')
        label_out = os.path.join(output_root, split, 'label')

        os.makedirs(a_out, exist_ok=True)
        os.makedirs(b_out, exist_ok=True)
        os.makedirs(label_out, exist_ok=True)

        s1_map = {normalize_name(f): f for f in os.listdir(s1_dir)}
        s2_map = {normalize_name(f): f for f in os.listdir(s2_dir)}
        label_map = {normalize_name(f): f for f in os.listdir(label_dir)}

        common_files_to_process = [
            (s1_map[n], s2_map[n], label_map[n])
            for n in sorted(list(common_normalized_names))
        ]

        for s1_file, s2_file, label_file in tqdm(common_files_to_process, desc=f"➡️  Standardizing {split}", ncols=100):
            try:
                s1_img = read_tiff_as_image(os.path.join(s1_dir, s1_file), size=image_size, rgb=True)
                s2_img = read_tiff_as_image(os.path.join(s2_dir, s2_file), size=image_size, rgb=True)
                label_img = read_tiff_as_image(os.path.join(label_dir, label_file), size=image_size, rgb=False, is_label=True)

                base_name = os.path.splitext(s1_file)[0] + ".png"
                s1_img.save(os.path.join(a_out, base_name))
                s2_img.save(os.path.join(b_out, base_name))
                label_img.save(os.path.join(label_out, base_name))

            except Exception as e:
                tqdm.write(f"[!] Skipped {s1_file} due to error: {e}")

        print(f"✅ Done: {len(common_files_to_process)} image triplets standardized for '{split}'.")

    print("\n🎉 DFC2020 standardization complete. Check the output directory.")

if __name__ == "__main__":
    original_dataset_path = r"data\DFC2020"
    output_path = r"data\standardized\dfc2020"
    os.makedirs(output_path, exist_ok=True)
    standardize_dfc2020(original_dataset_path, output_path, image_size=(256, 256))
