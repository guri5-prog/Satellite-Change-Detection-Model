import os
import cv2
import numpy as np

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

def standardize_dsifn(dsifn_root, out_root):
    sets = ['train', 'test']

    for split in sets:
        print(f"\n[→] Processing split: {split}")

        split_dir = os.path.join(dsifn_root, split)
        a_dir = os.path.join(split_dir, "A")
        b_dir = os.path.join(split_dir, "B")
        label_dir = os.path.join(split_dir, "label")

        out_a = os.path.join(out_root, split, "A")
        out_b = os.path.join(out_root, split, "B")
        out_mask = os.path.join(out_root, split, "mask")
        out_labels = os.path.join(out_root, split, "labels")

        os.makedirs(out_a, exist_ok=True)
        os.makedirs(out_b, exist_ok=True)
        os.makedirs(out_mask, exist_ok=True)
        os.makedirs(out_labels, exist_ok=True)

        a_images = sorted([f for f in os.listdir(a_dir) if f.endswith(('.jpg', '.png', '.tif'))])

        for img_name in a_images:
            try:
                img_a_path = os.path.join(a_dir, img_name)
                img_b_path = os.path.join(b_dir, img_name)
                mask_path = os.path.join(label_dir, img_name)

                if not (os.path.exists(img_a_path) and os.path.exists(img_b_path) and os.path.exists(mask_path)):
                    print(f"[!] Missing files for {img_name}, skipping...")
                    continue

                img_a = cv2.imread(img_a_path)
                img_b = cv2.imread(img_b_path)
                mask = cv2.imread(mask_path, 0)

                if img_a is None or img_b is None or mask is None:
                    print(f"[!] Could not read one or more files for {img_name}, skipping...")
                    continue

                cv2.imwrite(os.path.join(out_a, img_name), img_a)
                cv2.imwrite(os.path.join(out_b, img_name), img_b)
                cv2.imwrite(os.path.join(out_mask, img_name), mask)

                bboxes = mask_to_bboxes(mask)
                label_path = os.path.join(out_labels, img_name.rsplit('.', 1)[0] + ".txt")
                save_yolo_format(bboxes, img_b.shape, label_path)

                print(f"[✓] {split}/{img_name}: {len(bboxes)} boxes")

            except Exception as e:
                print(f"[✗] Error processing {img_name}: {e}")

if __name__ == "__main__":
    standardize_dsifn(
        dsifn_root=r"data\DSIFN Train Test",
        out_root=r"data/standardized/dsifn"
    )
