from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import os
from tqdm import tqdm
import torch
import json
import random


def clahe(tensor_image):
    tensor_image_int = tensor_image.mul(255).byte()
    hist = torch.histc(tensor_image_int, bins=256, min=0, max=255)
    cdf = hist.float().cumsum(0)
    cdf = cdf / cdf[-1]
    tensor_image_eq = torch.searchsorted(cdf, tensor_image_int.float())
    tensor_image_eq = tensor_image_eq / 255

    return tensor_image_eq


def crop(image, keypoints, perc_points=0.85, pad=5):
    norm_keypoints = MinMaxScaler().fit_transform(keypoints)
    total = len(keypoints)
    best_dist = 1
    best_clusters = None
    best_asm = None
    for eps in [0.01, 0.025, 0.05, 0.1, 0.2]:
        clusters = DBSCAN(eps=eps).fit_predict(norm_keypoints)
        counts = pd.Series(clusters).value_counts().sort_values(ascending=False)
        counts = counts[counts.index > -1]
        if len(counts) == 0:
            continue

        cumsums = np.cumsum(counts.values) / total
        dists = np.abs(cumsums - perc_points)
        best_ix = np.argmin(dists)

        if dists[best_ix] < best_dist:
            best_dist = dists[best_ix]
            best_clusters = list(counts.head(best_ix + 1).index)
            best_asm = clusters

    mask = np.isin(best_asm, best_clusters)

    miny = int(np.min(keypoints[mask][:, 1]))
    miny = max(miny - pad, 0)

    maxy = int(np.max(keypoints[mask][:, 1]))
    maxy = min(maxy + pad, image.shape[0])

    minx = int(np.min(keypoints[mask][:, 0]))
    minx = max(minx - pad, 0)

    maxx = int(np.max(keypoints[mask][:, 0]))
    maxx = min(maxx + pad, image.shape[1])

    # print(image.shape[0], image.shape[1], maxy + 1 - miny, maxx + 1 - minx)

    return miny, maxy + 1, minx, maxx + 1, minx, miny


def draw_picture(image1_path, image2_path, points0, points1, saved_name):
    # image1_path = image1_path.replace('-mask', '')
    # image2_path = image2_path.replace('-mask', '')
    #
    # if image1_path not in os.listdir(os.path.dirname(image1_path)):
    #     image1_path = image1_path.replace('png', 'jpg')
    # if image2_path not in os.listdir(os.path.dirname(image2_path)):
    #     image2_path = image2_path.replace('png', 'jpg')

    # if "png" in image1_path:
    #     image1_path = image1_path.replace('png', 'jpg')
    # if "png" in image2_path:
    #     image2_path = image2_path.replace('png', 'jpg')

    # print(image1_path, image2_path)

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    h1, w1, _ = image1.shape
    h2, w2, _ = image2.shape
    height = max(h1, h2)
    width = w1 + w2

    vis_image = np.zeros((height, width, 3), dtype=np.uint8)
    vis_image[:h1, :w1, :] = image1
    vis_image[:h2, w1:w1 + w2, :] = image2

    keypoints1 = points0
    keypoints2 = points1
    keypoints2[:, 0] += w1

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    ax.set_axis_off()

    for (x1, y1), (x2, y2) in zip(keypoints1, keypoints2):
        ax.scatter([x1, x2], [y1, y2], c='red', s=10)
        con = ConnectionPatch(xyA=(x2, y2), xyB=(x1, y1), coordsA="data", coordsB="data",
                              axesA=ax, axesB=ax, color="green", linewidth=2.5, linestyle='dotted')
        ax.add_artist(con)

    plt.savefig(saved_name, bbox_inches='tight')
    plt.close()


def extract_match(image1_path, image2_path, mask, extractor, matcher):
    image0 = load_image(image1_path).cuda()
    image1 = load_image(image2_path).cuda()

    # if image0.mean() < 1e-4:
    #     image0 = clahe(image0)
    # if image1.mean() < 1e-4:
    #     image1 = clahe(image1)

    xx, yy, ww, hh = cv2.boundingRect(mask[..., 0])
    image0 = image0[:, yy:yy + hh, xx:xx + ww]
    image1 = image1[:, yy:yy + hh, xx:xx + ww]

    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    matches = matches01['matches']
    points0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()
    points1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()

    points0[:, 0] += xx
    points0[:, 1] += yy

    points1[:, 0] += xx
    points1[:, 1] += yy

    return points0, points1


def extract_match_1(image1_path, image2_path, extractor, matcher):
    image0 = load_image(image1_path).cuda()
    image1 = load_image(image2_path).cuda()

    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    matches = matches01['matches']
    points0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()
    points1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()

    return points0, points1


def main(image1_path, image2_path, extractor, matcher, mask, saved_name):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    points1, points2 = extract_match(image1_path, image2_path, mask, extractor, matcher)

    # Fm, inliers = cv2.findFundamentalMat(points1, points2, cv2.USAC_MAGSAC, 1.5, 0.999, 100000)
    # inliers = inliers > 0
    # points1 = points1[inliers[:, 0], :]
    # points2 = points2[inliers[:, 0], :]

    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.erode(mask, structuring_element, iterations=2)

    remaining_points1 = []
    remaining_points2 = []
    for i, point in enumerate(points1):
        x, y = point
        x2, y2 = points2[i]
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            if np.any(mask[int(y), int(x)]) and np.any(mask[int(y2), int(x2)]) != 0:
                remaining_points1.append(point)
                remaining_points2.append(points2[i])
    points1 = np.array(remaining_points1)
    points2 = np.array(remaining_points2)

    if points1.shape[0] == 0:
        return 0

    scaled_points1 = points1.copy()
    scaled_points2 = points2.copy()

    scaled_points1[:, 0] = points1[:, 0] / image1.shape[1]
    scaled_points1[:, 1] = points1[:, 1] / image1.shape[0]

    scaled_points2[:, 0] = points2[:, 0] / image2.shape[1]
    scaled_points2[:, 1] = points2[:, 1] / image2.shape[0]

    distances = np.sqrt(
        (scaled_points2[:, 0] - scaled_points1[:, 0]) ** 2 + (scaled_points2[:, 1] - scaled_points1[:, 1]) ** 2)
    average_distance = np.round(np.mean(distances), 5)

    print("===================================")
    print(f"matches: {points1.shape[0]}, dist: {average_distance}")

    draw_picture(image1_path, image2_path, points1, points2, saved_name)

    image_path = saved_name
    image = cv2.imread(image_path)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 255)
    thickness = 2

    text = f"matches: {points1.shape[0]}, distance: {average_distance}"
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = image.shape[1] - text_width - 10
    text_y = image.shape[0] - 15

    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.imwrite(saved_name, image)

    return 1


def eval(image1_path, image2_path, extractor, matcher, mask, saved_name, test_name, match_thres, dist_thres):
    target_idx = 0
    folder_name = image1_path.split('/')[-2]
    idx1 = image1_path.split(folder_name + '/')[1].split(".")[0]
    idx2 = image2_path.split(folder_name + '/')[1].split(".")[0]

    if '-1-' in idx1:
        target_idx = int(idx2.split('-mask')[0].split('-')[1])
    if '-1-' in idx2:
        target_idx = int(idx1.split('-mask')[0].split('-')[1])

    found_lines = []
    with open(test_name, 'r') as file:
        for line in file:
            line_content = line.strip()
            if folder_name in line_content:
                elements = line_content.split()
                found_lines.extend(elements)
    is_same = int(found_lines[target_idx - 1])

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    points1, points2 = extract_match(image1_path, image2_path, mask, extractor, matcher)

    # Fm, inliers = cv2.findFundamentalMat(points1, points2, cv2.USAC_MAGSAC, 1.5, 0.999, 100000)
    # inliers = inliers > 0
    # points1 = points1[inliers[:, 0], :]
    # points2 = points2[inliers[:, 0], :]

    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.erode(mask, structuring_element, iterations=2)

    remaining_points1 = []
    remaining_points2 = []
    for i, point in enumerate(points1):
        x, y = point
        x2, y2 = points2[i]
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            if np.any(mask[int(y), int(x)]) and np.any(mask[int(y2), int(x2)]) != 0:
                remaining_points1.append(point)
                remaining_points2.append(points2[i])
    points1 = np.array(remaining_points1)
    points2 = np.array(remaining_points2)

    if points1.shape[0] == 0:
        pred = 0
        if pred != is_same:
            print(f"{os.path.basename(image1_path)} and {os.path.basename(image2_path)} matches failed !")
            return 0
        else:
            print(f"{os.path.basename(image1_path)} and {os.path.basename(image2_path)} matches successfully !")
            return 1

    scaled_points1 = points1.copy()
    scaled_points2 = points2.copy()

    scaled_points1[:, 0] = points1[:, 0] / image1.shape[1]
    scaled_points1[:, 1] = points1[:, 1] / image1.shape[0]

    scaled_points2[:, 0] = points2[:, 0] / image2.shape[1]
    scaled_points2[:, 1] = points2[:, 1] / image2.shape[0]

    distances = np.sqrt(
        (scaled_points2[:, 0] - scaled_points1[:, 0]) ** 2 + (scaled_points2[:, 1] - scaled_points1[:, 1]) ** 2)
    average_distance = np.round(np.mean(distances), 5)

    if points1.shape[0] > match_thres and average_distance < dist_thres:
        pred = 1
    else:
        pred = 0

    if pred != is_same:
        print(f"{os.path.basename(image1_path)} and {os.path.basename(image2_path)} matches failed !")
        print(f"matches: {points1.shape[0]}, dist: {average_distance}")
        return 0
    else:
        print(f"{os.path.basename(image1_path)} and {os.path.basename(image2_path)} matches successfully !")
        return 1

def main_gen_fold(image1_path, image2_path, extractor, matcher, saved_name):

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    points1, points2 = extract_match_1(image1_path, image2_path, extractor, matcher)

    # Fm, inliers = cv2.findFundamentalMat(points1, points2, cv2.USAC_MAGSAC, 1.5, 0.999, 100000)
    # inliers = inliers > 0
    # points1 = points1[inliers[:, 0], :]
    # points2 = points2[inliers[:, 0], :]

    # structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # mask = cv2.erode(mask, structuring_element, iterations=2)
    #
    # remaining_points1 = []
    # remaining_points2 = []
    # for i, point in enumerate(points1):
    #     x, y = point
    #     x2, y2 = points2[i]
    #     if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
    #         if np.any(mask[int(y), int(x)]) and np.any(mask[int(y2), int(x2)]) != 0:
    #             remaining_points1.append(point)
    #             remaining_points2.append(points2[i])
    # points1 = np.array(remaining_points1)
    # points2 = np.array(remaining_points2)

    if points1.shape[0] == 0:
        return 0

    scaled_points1 = points1.copy()
    scaled_points2 = points2.copy()

    scaled_points1[:, 0] = points1[:, 0] / image1.shape[1]
    scaled_points1[:, 1] = points1[:, 1] / image1.shape[0]

    scaled_points2[:, 0] = points2[:, 0] / image2.shape[1]
    scaled_points2[:, 1] = points2[:, 1] / image2.shape[0]

    distances = np.sqrt(
        (scaled_points2[:, 0] - scaled_points1[:, 0]) ** 2 + (scaled_points2[:, 1] - scaled_points1[:, 1]) ** 2)
    average_distance = np.round(np.mean(distances), 5)

    print("===================================")
    print(f"matches: {points1.shape[0]}, dist: {average_distance}")

    draw_picture(image1_path, image2_path, points1, points2, saved_name)

    image_path = saved_name
    image = cv2.imread(image_path)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 255)
    thickness = 2

    text = f"matches: {points1.shape[0]}, distance: {average_distance}"
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = image.shape[1] - text_width - 10
    text_y = image.shape[0] - 15

    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.imwrite(saved_name, image)

    return 1


def main_gen_fold_no_img(image1_path, image2_path, extractor, matcher):

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    points1, points2 = extract_match_1(image1_path, image2_path, extractor, matcher)

    # Fm, inliers = cv2.findFundamentalMat(points1, points2, cv2.USAC_MAGSAC, 1.5, 0.999, 100000)
    # inliers = inliers > 0
    # points1 = points1[inliers[:, 0], :]
    # points2 = points2[inliers[:, 0], :]

    # structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # mask = cv2.erode(mask, structuring_element, iterations=2)
    #
    # remaining_points1 = []
    # remaining_points2 = []
    # for i, point in enumerate(points1):
    #     x, y = point
    #     x2, y2 = points2[i]
    #     if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
    #         if np.any(mask[int(y), int(x)]) and np.any(mask[int(y2), int(x2)]) != 0:
    #             remaining_points1.append(point)
    #             remaining_points2.append(points2[i])
    # points1 = np.array(remaining_points1)
    # points2 = np.array(remaining_points2)

    if points1.shape[0] == 0:
        print(f"img1: {os.path.basename(image1_path)}, img2: {os.path.basename(image2_path)}, matches: 0, dist: 100")
        return 0, 100

    scaled_points1 = points1.copy()
    scaled_points2 = points2.copy()

    scaled_points1[:, 0] = points1[:, 0] / image1.shape[1]
    scaled_points1[:, 1] = points1[:, 1] / image1.shape[0]

    scaled_points2[:, 0] = points2[:, 0] / image2.shape[1]
    scaled_points2[:, 1] = points2[:, 1] / image2.shape[0]

    distances = np.sqrt(
        (scaled_points2[:, 0] - scaled_points1[:, 0]) ** 2 + (scaled_points2[:, 1] - scaled_points1[:, 1]) ** 2)
    average_distance = np.round(np.mean(distances), 5)

    print("===================================")
    print(f"img1: {os.path.basename(image1_path)}, img2: {os.path.basename(image2_path)}, matches: {points1.shape[0]}, dist: {average_distance}")

    return points1.shape[0], average_distance


if __name__ == '__main__':

    extractor = SuperPoint(max_num_keypoints=1024).eval().cuda()
    matcher = LightGlue(features='superpoint', depth_confidence=0.9, width_confidence=0.95).eval().cuda()

    def compute_similarity_for_squeezes(squeeze_dir, extractor, matcher):
        
        squeezes_list= []
        for squeeze_name in os.listdir(squeeze_dir):
            # Randomly select one representative image from each sequence.
            squeeze_images_list = os.listdir(os.path.join(squeeze_dir, squeeze_name))
            selected_image_path = os.path.join(os.path.join(squeeze_dir, squeeze_name), random.choice(squeeze_images_list))
            squeezes_list.append(selected_image_path)
        
        similarity_dict = {}
        for i, img1 in tqdm(enumerate(squeezes_list), total=len(squeezes_list), desc="Processing Images"):
            squeeze_name_1 = img1.split('/')[-2]
            similarity_dict[squeeze_name_1] = {}
            for j in range(i + 1, len(squeezes_list)):
                img2 = squeezes_list[j]
                squeeze_name_2 = img2.split('/')[-2]
                matcher_points, average_distance = main_gen_fold_no_img(img1, img2, extractor, matcher)
                similarity_dict[squeeze_name_1][squeeze_name_2] = {
                'matcher_points': float(matcher_points),
                'average_distance': float(average_distance)
            }

        return similarity_dict

    def save_similarity_dict(similarity_dict, save_path):
        with open(save_path, 'w') as json_file:
            json.dump(similarity_dict, json_file, indent=4)
        print(f"saved: {save_path}!")


    squeeze_dir = './Data/train'
    similarity_results = compute_similarity_for_squeezes(squeeze_dir, extractor, matcher)

    save_path = './Data/similarity_results.json'
    save_similarity_dict(similarity_results, save_path)


