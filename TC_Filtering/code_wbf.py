import json
import numpy as np
from collections import defaultdict
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm

def apply_wbf_to_json(json_path, output_path, method_weights, iou_thr=0.5, skip_box_thr=0.3, sigma=0.1):
    """
    Apply Weighted Boxes Fusion (WBF) on bounding boxes from different methods, with progress tracking.

    Args:
        json_path (str): Path to the input JSON file containing detected boxes.
        output_path (str): Path to save the WBF-processed JSON file.
        method_weights (dict): Weights for different methods, e.g., {"yolox": 2, "dino": 1}.
        iou_thr (float): IoU threshold for WBF.
        skip_box_thr (float): Minimum score threshold for WBF.
        sigma (float): Sigma value for Soft-NMS (not used in WBF but required by the function).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    processed_data = {}
    total_images = sum(len(images) for images in data.values())  

    with tqdm(total=total_images, desc="Processing WBF", unit="image") as pbar:
        for sequence, images in data.items():
            processed_data[sequence] = {}

            for image_name, detections in images.items():
                if not detections:  # If no detections, keep it empty
                    processed_data[sequence][image_name] = []
                    pbar.update(1)
                    continue

                # Group boxes by method
                method_boxes = defaultdict(list)
                method_scores = defaultdict(list)

                for det in detections:
                    x1, y1, x2, y2, score, method = det
                    method_boxes[method].append([x1, y1, x2, y2])
                    method_scores[method].append(score)

                # Prepare WBF input
                boxes_list, scores_list, labels_list, weights = [], [], [], []

                for method, boxes in method_boxes.items():
                    scores = method_scores[method]
                    boxes_list.append(np.array(boxes)) 
                    scores_list.append(np.array(scores))
                    labels_list.append(np.zeros(len(boxes), dtype=int))  # Single-class case
                    weights.append(method_weights.get(method, 1))  # Default weight is 1

                # Apply WBF only if multiple sources exist
                if len(boxes_list) > 1:
                    boxes, scores, labels = weighted_boxes_fusion(
                        boxes_list, scores_list, labels_list, weights=weights,
                        iou_thr=iou_thr, skip_box_thr=skip_box_thr
                    )
                else:  # If only one method, keep original detections
                    boxes, scores = boxes_list[0], scores_list[0]

                processed_data[sequence][image_name] = [
                    [float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score), "wbf_fused"]
                    for box, score in zip(boxes, scores)
                ] if len(boxes) > 0 else []

                pbar.update(1)  

    # Save results
    with open(output_path, "w") as f:
        json.dump(processed_data, f, indent=4)

    print(f"\nWBF applied and saved to {output_path}")

# Example usage
json_path = "./TC_Filtering/processed_merged_output.json"
output_path = "./TC_Filtering/wbf_output.json"
method_weights = {
    "yolox_afpn_p2345_c3k2h_imgsz640_diff_fold1": 1,
    "yolox_afpn_p2345_c3k2h_imgsz640_diff_fold2": 1,
}

apply_wbf_to_json(json_path, output_path, method_weights)
