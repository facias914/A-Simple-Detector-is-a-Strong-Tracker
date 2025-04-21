# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import shutil
import re

# ��������Ŀ¼
input_dir = "/mnt/data1/pcx/yolov11/cvpr/UAV/data/test2/"
output_dir = "/mnt/data1/pcx/yolov11/cvpr/UAV/data/test2_newdata/"

# �������Ŀ¼
os.makedirs(output_dir, exist_ok=True)

# ��ȡ�������ļ���
folders = sorted([f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))])

# ����ÿ���ļ���
for folder in folders:
    folder_path = os.path.join(input_dir, folder)
    output_folder_path = os.path.join(output_dir, folder)

    # ��������ļ���
    os.makedirs(output_folder_path, exist_ok=True)

    # ��ȡ���ļ����е�����ͼ�񣨰���ֵ����
    images = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))],
        key=lambda x: int(re.match(r'^(\d+)', x).group(1))  # ���ļ����е���ֵ����
    )

    # ��¼ǰ��֡�ĻҶ���Ϣ
    prev_avg = None
    prev_prev_avg = None
    prev_frame_idx = None
    prev_prev_frame_idx = None

    # ����ͼ��
    for img_name in images:
        current_img_path = os.path.join(folder_path, img_name)
        
        # ��ȡ��ǰ֡���
        current_frame_idx = int(re.match(r'^(\d+)', img_name).group(1))

        # ===== �����޸Ĳ��� =====
        # ��һ֡����ʼ��ǰ֡��Ϣ��
        if prev_frame_idx is None:
            # ��ȡͼ�񲢼���ҶȾ�ֵ
            img = cv2.imread(current_img_path)
            if img is None:
                print(f"wufaduqu {current_img_path}")
                continue
            prev_avg = np.mean(img, axis=2)
            prev_frame_idx = current_frame_idx
            continue

        # �ڶ�֡����ʼ��ǰǰ֡��Ϣ��
        if prev_prev_frame_idx is None:
            # ���������
            if current_frame_idx != prev_frame_idx + 1:
                print(f"zhenbulianxu {prev_frame_idx} -> {current_frame_idx}")
                prev_frame_idx = None
                prev_avg = None
                continue

            # ��ȡͼ�񲢼���ҶȾ�ֵ
            img = cv2.imread(current_img_path)
            if img is None:
                print(f"wufaduqu {current_img_path}")
                continue
            prev_prev_avg = prev_avg  # �����һ֡�ĻҶ�
            prev_avg = np.mean(img, axis=2)  # ����Ϊ�ڶ�֡�ĻҶ�
            prev_prev_frame_idx = prev_frame_idx
            prev_frame_idx = current_frame_idx
            continue

        # ��������ԣ���ǰ֡�Ƿ����ǰһ֡+1��
        if current_frame_idx != prev_frame_idx + 1:
            print(f"zhenbulianxu {prev_frame_idx} -> {current_frame_idx}")
            prev_prev_avg = None
            prev_avg = None
            prev_prev_frame_idx = None
            prev_frame_idx = None
            continue

        # ===== ��������֡ =====
        # ��ȡ��ǰ֡
        current_img = cv2.imread(current_img_path)
        if current_img is None:
            print(f"wufaduqu {current_img_path}")
            continue

        # ���㵱ǰ֡�ҶȾ�ֵ
        current_avg = np.mean(current_img, axis=2)

        # ���������������Ҫ����
        flow = cv2.calcOpticalFlowFarneback(prev_avg, current_avg, None, 0.5, 5, 9, 5, 5, 1.1, 0) 
        flow_u = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow_v = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)

        # �ϲ���ͨ��
        merged = np.stack([current_avg, flow_v, flow_u], axis=-1).astype(np.uint8)

        # ������
        output_path = os.path.join(output_folder_path, img_name)
        cv2.imwrite(output_path, merged)

        # ���Ʊ�ǩ�ļ�
        txt_name = re.sub(r'\.(jpg|png)$', '.txt', img_name)
        src_txt = os.path.join(folder_path, txt_name)
        dst_txt = os.path.join(output_folder_path, txt_name)
        if os.path.exists(src_txt):
            shutil.copy(src_txt, dst_txt)

        # ����ǰ��֡��Ϣ
        prev_prev_avg = prev_avg
        prev_avg = current_avg
        prev_prev_frame_idx = prev_frame_idx
        prev_frame_idx = current_frame_idx