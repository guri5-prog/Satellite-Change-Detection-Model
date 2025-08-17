import os
import cv2
import numpy as np
from tqdm import tqdm

def mask_to_bboxes(mask, min_area=100):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= min_area:
            bboxes.append((x, y, w, h))
    return bboxes

def save_yolo_format(bboxes, img_shape, file_path, class_id=0):
    h, w = img_shape[:2]
    with open(file_path, 'w') as f:
        for x, y, bw, bh in bboxes:
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            bw_norm = bw / w
            bh_norm = bh / h
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")

def standardize_dfc2020(dfc_root, out_root):
    splits = ['train', 'val', 'test']

    for split in splits:
        print(f"\n🔄 Processing split: {split}")

        a_dir = os.path.join(dfc_root, split, "s1")
        b_dir = os.path.join(dfc_root, split, "s2")
        label_dir = os.path.join(dfc_root, split, "lc")

        out_a = os.path.join(out_root, split, "A")
        out_b = os.path.join(out_root, split, "B")
        out_mask = os.path.join(out_root, split, "mask")
        out_labels = os.path.join(out_root, split, "labels")

        os.makedirs(out_a, exist_ok=True)
        os.makedirs(out_b, exist_ok=True)
        os.makedirs(out_mask, exist_ok=True)
        os.makedirs(out_labels, exist_ok=True)

        image_names = sorted([
                         f for f in os.listdir(a_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
                        ])


        for img_name in tqdm(image_names, desc=f"➡️  {split}"):
            try:
                img_a_path = os.path.join(a_dir, img_name)
                img_b_path = os.path.join(b_dir, img_name)
                mask_path = os.path.join(label_dir, img_name)

                if not (os.path.exists(img_a_path) and os.path.exists(img_b_path) and os.path.exists(mask_path)):
                    print(f"[!] Skipping {img_name} due to missing file.")
                    continue

                img_a = cv2.imread(img_a_path)
                img_b = cv2.imread(img_b_path)
                mask = cv2.imread(mask_path, 0)  # grayscale

                if img_a is None or img_b is None or mask is None:
                    print(f"[!] Skipping {img_name} due to read error.")
                    continue

                cv2.imwrite(os.path.join(out_a, img_name), img_a)
                cv2.imwrite(os.path.join(out_b, img_name), img_b)
                cv2.imwrite(os.path.join(out_mask, img_name), mask)

                bboxes = mask_to_bboxes(mask)
                label_file = os.path.join(out_labels, img_name.rsplit('.', 1)[0] + ".txt")
                save_yolo_format(bboxes, img_b.shape, label_file)

            except Exception as e:
                print(f"[✗] Error processing {img_name}: {e}")

    print("\n✅ DFC2020 standardization complete.")

if __name__ == "__main__":
    standardize_dfc2020(
        dfc_root=r"data\DFC2020",
        out_root=r"data\standardized\dfc2020"
    )
