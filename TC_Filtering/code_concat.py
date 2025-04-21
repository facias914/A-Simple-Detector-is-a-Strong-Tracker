import json
import cv2
import os
from collections import defaultdict
from tqdm import tqdm


def reformat_json(input_json_path, max_bboxes, score_threshold):
    """
    Convert JSON with 'images/annotations' structure to a unified format.
    """
    model_name = os.path.basename(input_json_path).split('.')[0]

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    reformatted_data = defaultdict(lambda: defaultdict(list))
    image_info = {img["id"]: (img["file_name"].split("/")[-2:], img["width"], img["height"]) for img in data["images"]}

    for ann in tqdm(data["annotations"], desc=f"Processing {input_json_path}", unit="ann"):
        img_id, bbox, score = ann["image_id"], ann["bbox"], ann["score"]
        (folder, filename), width, height = image_info[img_id]

        if score >= score_threshold:
            x1, y1, w, h = bbox
            bbox_norm = [x1 / width, y1 / height, (x1 + w) / width, (y1 + h) / height, score, model_name]
            reformatted_data[folder][filename].append(bbox_norm)

    return _filter_top_bboxes(reformatted_data, max_bboxes)


def process_standard_json(json_path, max_bboxes, score_threshold):
    """
    Process JSON where bbox format is [cx, cy, w, h] (center format).
    """
    model_name = os.path.basename(json_path).split('.')[0]
    processed_data = defaultdict(lambda: defaultdict(list))

    with open(json_path, "r") as f:
        data = json.load(f)

    for item in tqdm(data, desc=f"Processing {json_path}", unit="item"):
        cx, cy, w, h = item["bbox"]
        score = item["score"]
        img_path = item["image_id"]
        img = cv2.VideoCapture(img_path)
        width, height = int(img.get(cv2.CAP_PROP_FRAME_WIDTH)), int(img.get(cv2.CAP_PROP_FRAME_HEIGHT))
        img.release()

        if score < score_threshold:
            continue

        x1, y1, x2, y2 = cx / width, cy / height, (cx + w) / width, (cy + h) / height

        folder, filename = item["image_id"].split("/")[-2:]
        processed_data[folder][filename].append([x1, y1, x2, y2, score, model_name])

    return _filter_top_bboxes(processed_data, max_bboxes)



def _filter_top_bboxes(data, max_bboxes):
    """
    Keep only the top max_bboxes per image based on confidence score.
    """
    for folder in data:
        for filename in data[folder]:
            data[folder][filename] = sorted(data[folder][filename], key=lambda x: x[4], reverse=True)[:max_bboxes]
    return data


def merge_jsons(json_configs, valid_txt_path, output_json_path):
    """
    Merge multiple JSON detections while respecting individual thresholds and max bbox counts.
    """
    with open(valid_txt_path, "r") as f:
        valid_images = defaultdict(list)
        for line in f:
            folder, filename = line.strip().split("/")[-2:]
            valid_images[folder].append(filename)

    detections = defaultdict(lambda: defaultdict(list))

    for method_name, config in tqdm(json_configs.items(), desc="Processing JSON files", unit="file"):
        json_path = config["path"]
        max_bboxes, score_threshold = config["MAX_BBOX_PER_IMAGE"], config["CONFIDENCE_THRESHOLD"]

        with open(json_path, "r") as f:
            json_data = json.load(f)

        processed_data = (
            reformat_json(json_path, max_bboxes, score_threshold) if "images" in json_data else
            process_standard_json(json_path, max_bboxes, score_threshold)
        )

        for folder, images in processed_data.items():
            for filename, bboxes in images.items():
                detections[folder][filename].extend(bboxes)

    final_output = _apply_per_json_limits(detections, json_configs, valid_images)

    with open(output_json_path, "w") as f:
        json.dump(final_output, f, indent=4)

    print(f"Merged JSON saved at {output_json_path}")


def _apply_per_json_limits(detections, json_configs, valid_images):
    """
    Apply max bbox constraints per JSON source per image.
    """
    final_output = {folder: {} for folder in valid_images}

    for folder, images in valid_images.items():
        for filename in images:
            if filename in detections[folder]:
                bbox_list = sorted(detections[folder][filename], key=lambda x: x[4], reverse=True)

                per_method_count = {method: 0 for method in json_configs}
                filtered_bboxes = []

                for bbox in bbox_list:
                    method = bbox[5]
                    if per_method_count[method] < json_configs[method]["MAX_BBOX_PER_IMAGE"]:
                        filtered_bboxes.append(bbox)
                        per_method_count[method] += 1

                final_output[folder][filename] = filtered_bboxes
            else:
                final_output[folder][filename] = []

    return final_output


# Example Usage
json_paths = {
    "yolox_afpn_p2345_c3k2h_imgsz640_diff_fold1": {
        "path": "./TC_Filtering/results/yolox_afpn_p2345_c3k2h_imgsz640_diff_fold1.json",
        "CONFIDENCE_THRESHOLD": 0.0,
        "MAX_BBOX_PER_IMAGE": 3
    },
    "yolox_afpn_p2345_c3k2h_imgsz640_diff_fold2": {
        "path": "./TC_Filtering/results/yolox_afpn_p2345_c3k2h_imgsz640_diff_fold2.json",
        "CONFIDENCE_THRESHOLD": 0.0,
        "MAX_BBOX_PER_IMAGE": 3
    },
}

valid_txt_path = "./TC_Filtering/results/test2.txt"
output_json_path = "./TC_Filtering/processed_merged_output.json"

merge_jsons(json_paths, valid_txt_path, output_json_path)