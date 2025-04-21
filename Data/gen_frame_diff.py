import os
import cv2
import numpy as np
import shutil
import re


input_dir = "./Data/train/"
output_dir = "./Data/fd_train/"  # "./Data/of_train/"


os.makedirs(output_dir, exist_ok=True)

folders = sorted([f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))])

for folder in folders:
    folder_path = os.path.join(input_dir, folder)
    output_folder_path = os.path.join(output_dir, folder)

    os.makedirs(output_folder_path, exist_ok=True)

    images = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))],
        key=lambda x: int(re.match(r'^(\d+)', x).group(1)) 
    )

    prev_avg = None
    prev_prev_avg = None
    prev_frame_idx = None
    prev_prev_frame_idx = None

    for img_name in images:
        current_img_path = os.path.join(folder_path, img_name)
        current_frame_idx = int(re.match(r'^(\d+)', img_name).group(1))

        if prev_frame_idx is None:
            img = cv2.imread(current_img_path)
            if img is None:
                print(f"wufaduqu {current_img_path}")
                continue
            prev_avg = np.mean(img, axis=2)
            prev_frame_idx = current_frame_idx
            continue

        if prev_prev_frame_idx is None:
            if current_frame_idx != prev_frame_idx + 1:
                print(f"zhenbulianxu {prev_frame_idx} -> {current_frame_idx}")
                prev_frame_idx = None
                prev_avg = None
                continue

            img = cv2.imread(current_img_path)
            if img is None:
                print(f"wufaduqu {current_img_path}")
                continue
            prev_prev_avg = prev_avg
            prev_avg = np.mean(img, axis=2) 
            prev_prev_frame_idx = prev_frame_idx
            prev_frame_idx = current_frame_idx
            continue

        if current_frame_idx != prev_frame_idx + 1:
            print(f"zhenbulianxu {prev_frame_idx} -> {current_frame_idx}")
            prev_prev_avg = None
            prev_avg = None
            prev_prev_frame_idx = None
            prev_frame_idx = None
            continue

        current_img = cv2.imread(current_img_path)
        if current_img is None:
            print(f"wufaduqu {current_img_path}")
            continue

        current_avg = np.mean(current_img, axis=2)
        diff1 = np.abs(current_avg - prev_avg)
        diff2 = np.abs(current_avg - prev_prev_avg)

        merged = np.stack([current_avg, diff1, diff2], axis=-1).astype(np.uint8)

        output_path = os.path.join(output_folder_path, img_name)
        cv2.imwrite(output_path, merged)

        txt_name = re.sub(r'\.(jpg|png)$', '.txt', img_name)
        src_txt = os.path.join(folder_path, txt_name)
        dst_txt = os.path.join(output_folder_path, txt_name)
        if os.path.exists(src_txt):
            shutil.copy(src_txt, dst_txt)

        prev_prev_avg = prev_avg
        prev_avg = current_avg
        prev_prev_frame_idx = prev_frame_idx
        prev_frame_idx = current_frame_idx