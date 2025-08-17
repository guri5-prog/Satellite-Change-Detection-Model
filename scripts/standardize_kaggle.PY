import os
import shutil
import cv2
import numpy as np

def create_change_mask(label1, label2):
    return (label1 != label2).astype(np.uint8) * 255

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

def standardize_kaggle_alt(input_root, out_root):
    im1_dir = os.path.join(input_root, "im1")
    im2_dir = os.path.join(input_root, "im2")
    label1_dir = os.path.join(input_root, "label1")
    label2_dir = os.path.join(input_root, "label2")

    os.makedirs(f"{out_root}/A", exist_ok=True)
    os.makedirs(f"{out_root}/B", exist_ok=True)
    os.makedirs(f"{out_root}/mask", exist_ok=True)
    os.makedirs(f"{out_root}/labels", exist_ok=True)

    files = sorted(os.listdir(im1_dir))
    for file in files:
        if not file.endswith(".png"):
            continue

        id = file.replace('.png', '')

        img_a_path = os.path.join(im1_dir, file)
        img_b_path = os.path.join(im2_dir, file)
        label1_path = os.path.join(label1_dir, file)
        label2_path = os.path.join(label2_dir, file)

        # Copy A & B images
        shutil.copy(img_a_path, os.path.join(out_root, "A", file))
        shutil.copy(img_b_path, os.path.join(out_root, "B", file))

        # Load label images
        lbl1 = cv2.imread(label1_path, 0)
        lbl2 = cv2.imread(label2_path, 0)

        # Compute binary change mask
        change_mask = create_change_mask(lbl1, lbl2)
        cv2.imwrite(os.path.join(out_root, "mask", file), change_mask)

        # Convert to YOLO bounding boxes
        img_b = cv2.imread(img_b_path)
        bboxes = mask_to_bboxes(change_mask)
        save_yolo_format(bboxes, img_b.shape, os.path.join(out_root, "labels", id + ".txt"))

        print(f"[✓] {file} → {len(bboxes)} boxes")

if __name__ == "__main__":
    standardize_kaggle_alt(
        input_root="data/kaggle_change", 
        out_root="data/standardized/kaggle"
    )
