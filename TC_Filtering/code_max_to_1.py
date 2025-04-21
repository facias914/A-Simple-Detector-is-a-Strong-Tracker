import json
from tqdm import tqdm

def filter_wbf_results(json_path, output_path, conf_threshold=0.1):
    """
    Filter WBF results by selecting the highest-confidence bounding box per image
    and ensuring it meets the confidence threshold.

    Args:
        json_path (str): Path to the input WBF JSON file.
        output_path (str): Path to save the filtered JSON file.
        conf_threshold (float): Minimum confidence score to keep a detection.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    filtered_data = {}
    total_images = sum(len(images) for images in data.values()) 

    with tqdm(total=total_images, desc="Filtering WBF results", unit="image") as pbar:
        for sequence, images in data.items():
            filtered_data[sequence] = {}

            for image_name, detections in images.items():
                if not detections:
                    filtered_data[sequence][image_name] = []
                    pbar.update(1)
                    continue

                best_box = max(detections, key=lambda x: x[4])  

                if best_box[4] > conf_threshold:
                    filtered_data[sequence][image_name] = [best_box]
                else:
                    filtered_data[sequence][image_name] = []

                pbar.update(1)

    with open(output_path, "w") as f:
        json.dump(filtered_data, f, indent=4)

    print(f"\nFiltered WBF results saved to {output_path}")

# Example usage
wbf_json_path = "./TC_Filtering/wbf_output.json"
filtered_output_path = "./TC_Filtering/wbf_filtered_output.json"

filter_wbf_results(wbf_json_path, filtered_output_path)
