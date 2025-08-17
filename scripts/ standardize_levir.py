import os
import cv2
import numpy as np
import shutil

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

def standardize_levir(levir_dir, out_dir):
    os.makedirs(f"{out_dir}/A", exist_ok=True)
    os.makedirs(f"{out_dir}/B", exist_ok=True)
    os.makedirs(f"{out_dir}/mask", exist_ok=True)
    os.makedirs(f"{out_dir}/labels", exist_ok=True)

    a_dir = os.path.join(levir_dir, "train/A")
    b_dir = os.path.join(levir_dir, "train/B")
    mask_dir = os.path.join(levir_dir, "train/label")

    for file in os.listdir(mask_dir):
        if not file.endswith(".png"):
            continue

        img_a = os.path.join(a_dir, file)
        img_b = os.path.join(b_dir, file)
        mask_path = os.path.join(mask_dir, file)

        shutil.copy(img_a, os.path.join(out_dir, "A", file))
        shutil.copy(img_b, os.path.join(out_dir, "B", file))
        shutil.copy(mask_path, os.path.join(out_dir, "mask", file))

        mask = cv2.imread(mask_path, 0)
        img = cv2.imread(img_b)
        bboxes = mask_to_bboxes(mask)
        yolo_label_path = os.path.join(out_dir, "labels", file.replace('.png', '.txt'))
        save_yolo_format(bboxes, img.shape, yolo_label_path)
        print(f"[✓] Processed {file}, Boxes: {len(bboxes)}")

# Run the script
if __name__ == "__main__":
    standardize_levir(
        levir_dir="data/LEVIR CD",
        out_dir="data/standardized/levir"
    )
