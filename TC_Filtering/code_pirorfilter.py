import json
import os

MAX_MOVEMENT = 0.09  
IMAGE_SIZE = 512 


json_path = "./TC_Filtering/processed_merged_output.json"
with open(json_path, 'r') as f:
    data = json.load(f)

def get_center_and_size(box):

    x, y, w, h = box[:4]
    center_x = x + w / 2
    center_y = y + h / 2
    return center_x, center_y, w, h

def filter_unrealistic_boxes(video_data):

    prev_center = None  
    filtered_video_data = {}

    for frame_id, boxes in sorted(video_data.items()): 
        if not boxes or boxes == [[]]: 
            filtered_video_data[frame_id] = [[]]
            prev_center = None
            continue

        valid_boxes = []
        for box in boxes:
            center_x, center_y, _, _ = get_center_and_size(box)

            if prev_center is None:
                valid_boxes.append(box)
                prev_center = (center_x, center_y)
                continue

            movement = ((center_x - prev_center[0]) ** 2 + (center_y - prev_center[1]) ** 2) ** 0.5
            if movement <= MAX_MOVEMENT:
                valid_boxes.append(box)

        filtered_video_data[frame_id] = valid_boxes if valid_boxes else [[]]

        if valid_boxes:
            prev_center = get_center_and_size(valid_boxes[0])[:2]

    return filtered_video_data

def interpolate_missing_frames(video_data):

    frames = sorted(video_data.keys())
    filled_video_data = video_data.copy()

    for i in range(1, len(frames) - 1): 
        if video_data[frames[i]] != [[]]:
            continue

        prev_idx = i - 1
        next_idx = i + 1
        while prev_idx >= 0 and video_data[frames[prev_idx]] == [[]]:
            prev_idx -= 1
        while next_idx < len(frames) and video_data[frames[next_idx]] == [[]]:
            next_idx += 1

        if prev_idx >= 0 and next_idx < len(frames):
            prev_box = video_data[frames[prev_idx]][0]
            next_box = video_data[frames[next_idx]][0]

            prev_center = get_center_and_size(prev_box)
            next_center = get_center_and_size(next_box)

            alpha = (i - prev_idx) / (next_idx - prev_idx)
            interpolated_box = [
                prev_center[j] * (1 - alpha) + next_center[j] * alpha
                for j in range(4)
            ] + [0.5, "interpolated"] 

            filled_video_data[frames[i]] = [interpolated_box]

        elif prev_idx >= 0:
            filled_video_data[frames[i]] = video_data[frames[prev_idx]]

    return filled_video_data

processed_data = {}
for video_name, video_data in data.items():
    print(f"Squeeze {video_name} ...")

    filtered_video_data = filter_unrealistic_boxes(video_data)

    completed_video_data = interpolate_missing_frames(filtered_video_data)

    processed_data[video_name] = completed_video_data

output_path = "./TC_Filtering/processed_merged_output.json"
with open(output_path, 'w') as f:
    json.dump(processed_data, f, indent=2)

print(f"Save {output_path}")
