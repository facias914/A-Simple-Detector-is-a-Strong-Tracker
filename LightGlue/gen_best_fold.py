import os
import random
import json
import re
from collections import defaultdict

def sort_images_by_number(image_list):

    def extract_number(image_name):
        return int(re.findall(r'\d+', image_name)[0])
    
    return sorted(image_list, key=extract_number)

def load_similarity_dict(similarity_path):

    with open(similarity_path, 'r') as json_file:
        similarity_dict = json.load(json_file)
    return similarity_dict

def group_images_by_similarity(img_list, similarity_dict, threshold=100):

    groups = []  
    visited = set()  
    image_graph = defaultdict(set)
    
    for img in img_list:
        if img in similarity_dict:
            for similar_img, metrics in similarity_dict[img].items():
                if metrics['matcher_points'] > threshold:
                    image_graph[img].add(similar_img)
                    image_graph[similar_img].add(img)  
    
    def dfs(img, group):
        stack = [img]
        while stack:
            current_img = stack.pop()
            if current_img not in visited:
                visited.add(current_img)
                group.add(current_img)
                stack.extend(image_graph[current_img] - visited)
    
    for img in img_list:
        if img not in visited:
            group = set()
            dfs(img, group)
            groups.append(list(group))
    
    return groups

def split_groups_into_folds_greedy(groups, num_folds=5):

    groups = sorted(groups, key=lambda x: len(x), reverse=True)  # 按组大小降序排序
    folds = [[] for _ in range(num_folds)]
    fold_sizes = [0] * num_folds
    
    for group in groups:

        min_fold_idx = fold_sizes.index(min(fold_sizes))
        folds[min_fold_idx].extend(group)
        fold_sizes[min_fold_idx] += len(group)
    
    return folds

def calculate_val_difference(folds):

    val_sizes = [len(fold) for fold in folds]
    return max(val_sizes) - min(val_sizes)

def save_train_val_folds(folds, save_dir, img_dir):

    num_folds = len(folds)
    for fold_idx in range(num_folds):
        val_fold = folds[fold_idx]
        train_fold = [img for i, fold in enumerate(folds) if i != fold_idx for img in fold]
        
        train_fold = sort_images_by_number(train_fold)
        val_fold = sort_images_by_number(val_fold)
        
        train_fold = [os.path.join(img_dir, img) for img in train_fold]
        val_fold = [os.path.join(img_dir, img) for img in val_fold]
        
        train_file = os.path.join(save_dir, f"train_full_fold_{fold_idx+1}.txt")
        val_file = os.path.join(save_dir, f"valid_full_fold_{fold_idx+1}.txt")
        
        with open(train_file, 'w') as f_train:
            for img in train_fold:
                tt = os.path.basename(img).split(".jpg")[0]
                f_train.write(f"{tt}\n")
        
        with open(val_file, 'w') as f_val:
            for img in val_fold:
                zz = os.path.basename(img).split(".jpg")[0]
                f_val.write(f"{zz}\n")

def main():

    similarity_path = "./Data/similarity_results.json"
    squeeze_dir = './Data/train/'
    save_dir = './Data/best_folds/'

    os.makedirs(save_dir, exist_ok=True)
    
    similarity_dict = load_similarity_dict(similarity_path)
    squeeze_list = os.listdir(squeeze_dir)
    
    groups = group_images_by_similarity(squeeze_list, similarity_dict, threshold=100)
    
    folds = split_groups_into_folds_greedy(groups, num_folds=5)
    
    val_diff = calculate_val_difference(folds)
    print(f"Validation difference: {val_diff}")
    
    save_train_val_folds(folds, save_dir, squeeze_dir)
    print(f"Folds saved to {save_dir}")

if __name__ == "__main__":
    main()