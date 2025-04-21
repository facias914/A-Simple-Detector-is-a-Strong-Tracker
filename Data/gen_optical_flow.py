# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import shutil
import re

# 输入和输出目录
input_dir = "/mnt/data1/pcx/yolov11/cvpr/UAV/data/test2/"
output_dir = "/mnt/data1/pcx/yolov11/cvpr/UAV/data/test2_newdata/"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 获取所有子文件夹
folders = sorted([f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))])

# 遍历每个文件夹
for folder in folders:
    folder_path = os.path.join(input_dir, folder)
    output_folder_path = os.path.join(output_dir, folder)

    # 创建输出文件夹
    os.makedirs(output_folder_path, exist_ok=True)

    # 获取该文件夹中的所有图像（按数值排序）
    images = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))],
        key=lambda x: int(re.match(r'^(\d+)', x).group(1))  # 按文件名中的数值排序
    )

    # 记录前两帧的灰度信息
    prev_avg = None
    prev_prev_avg = None
    prev_frame_idx = None
    prev_prev_frame_idx = None

    # 处理图像
    for img_name in images:
        current_img_path = os.path.join(folder_path, img_name)
        
        # 提取当前帧序号
        current_frame_idx = int(re.match(r'^(\d+)', img_name).group(1))

        # ===== 核心修改部分 =====
        # 第一帧（初始化前帧信息）
        if prev_frame_idx is None:
            # 读取图像并计算灰度均值
            img = cv2.imread(current_img_path)
            if img is None:
                print(f"wufaduqu {current_img_path}")
                continue
            prev_avg = np.mean(img, axis=2)
            prev_frame_idx = current_frame_idx
            continue

        # 第二帧（初始化前前帧信息）
        if prev_prev_frame_idx is None:
            # 检查连续性
            if current_frame_idx != prev_frame_idx + 1:
                print(f"zhenbulianxu {prev_frame_idx} -> {current_frame_idx}")
                prev_frame_idx = None
                prev_avg = None
                continue

            # 读取图像并计算灰度均值
            img = cv2.imread(current_img_path)
            if img is None:
                print(f"wufaduqu {current_img_path}")
                continue
            prev_prev_avg = prev_avg  # 保存第一帧的灰度
            prev_avg = np.mean(img, axis=2)  # 更新为第二帧的灰度
            prev_prev_frame_idx = prev_frame_idx
            prev_frame_idx = current_frame_idx
            continue

        # 检查连续性（当前帧是否等于前一帧+1）
        if current_frame_idx != prev_frame_idx + 1:
            print(f"zhenbulianxu {prev_frame_idx} -> {current_frame_idx}")
            prev_prev_avg = None
            prev_avg = None
            prev_prev_frame_idx = None
            prev_frame_idx = None
            continue

        # ===== 正常处理帧 =====
        # 读取当前帧
        current_img = cv2.imread(current_img_path)
        if current_img is None:
            print(f"wufaduqu {current_img_path}")
            continue

        # 计算当前帧灰度均值
        current_avg = np.mean(current_img, axis=2)

        # 计算光流，参数需要调节
        flow = cv2.calcOpticalFlowFarneback(prev_avg, current_avg, None, 0.5, 5, 9, 5, 5, 1.1, 0) 
        flow_u = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow_v = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)

        # 合并三通道
        merged = np.stack([current_avg, flow_v, flow_u], axis=-1).astype(np.uint8)

        # 保存结果
        output_path = os.path.join(output_folder_path, img_name)
        cv2.imwrite(output_path, merged)

        # 复制标签文件
        txt_name = re.sub(r'\.(jpg|png)$', '.txt', img_name)
        src_txt = os.path.join(folder_path, txt_name)
        dst_txt = os.path.join(output_folder_path, txt_name)
        if os.path.exists(src_txt):
            shutil.copy(src_txt, dst_txt)

        # 更新前两帧信息
        prev_prev_avg = prev_avg
        prev_avg = current_avg
        prev_prev_frame_idx = prev_frame_idx
        prev_frame_idx = current_frame_idx