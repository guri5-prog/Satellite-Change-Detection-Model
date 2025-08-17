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

def standardize_oscd_fixed(oscd_root, mask_root, out_root):
    os.makedirs(f"{out_root}/A", exist_ok=True)
    os.makedirs(f"{out_root}/B", exist_ok=True)
    os.makedirs(f"{out_root}/mask", exist_ok=True)
    os.makedirs(f"{out_root}/labels", exist_ok=True)

    cities = [c for c in os.listdir(oscd_root) if os.path.isdir(os.path.join(oscd_root, c))]

    for city in cities:
        city_dir = os.path.join(oscd_root, city)
        a_dir = os.path.join(city_dir, "imgs_1_rect")
        b_dir = os.path.join(city_dir, "imgs_2_rect")

        # Example: dataset/oscd/oscd train/abudhabi/cm/cm.png
        mask_path = os.path.join(mask_root, city, "cm", "cm.png")
        if not os.path.exists(mask_path):
            print(f"[!] Mask not found for {city}, skipping...")
            continue

        print(f"\n[→] Processing city: {city}")
        try:
            a_files = sorted([f for f in os.listdir(a_dir) if f.endswith('.tif')])
            b_files = sorted([f for f in os.listdir(b_dir) if f.endswith('.tif')])

            if len(a_files) < 3 or len(b_files) < 3:
                print(f"[!] Less than 3 bands found in {city}, skipping...")
                continue

            bands_a = [cv2.imread(os.path.join(a_dir, tif), cv2.IMREAD_GRAYSCALE) for tif in a_files[:3]]
            bands_b = [cv2.imread(os.path.join(b_dir, tif), cv2.IMREAD_GRAYSCALE) for tif in b_files[:3]]

            if any(b is None for b in bands_a) or any(b is None for b in bands_b):
                print(f"[!] Failed to read one or more bands for {city}, skipping...")
                continue

            img_a = cv2.merge(bands_a)
            img_b = cv2.merge(bands_b)
            img_name = f"{city}.png"

            cv2.imwrite(os.path.join(out_root, "A", img_name), img_a)
            cv2.imwrite(os.path.join(out_root, "B", img_name), img_b)

            mask = cv2.imread(mask_path, 0)
            cv2.imwrite(os.path.join(out_root, "mask", img_name), mask)

            bboxes = mask_to_bboxes(mask)
            save_yolo_format(bboxes, img_b.shape, os.path.join(out_root, "labels", img_name.replace('.png', '.txt')))

            print(f"[✓] {city}: {len(bboxes)} boxes")

        except Exception as e:
            print(f"[✗] Failed to process {city}: {e}")

if __name__ == "__main__":
    standardize_oscd_fixed(
        oscd_root=r"data/oscd",
        mask_root=r"data/oscd/oscd train",
        out_root=r"data/standardized/oscd"
    )
