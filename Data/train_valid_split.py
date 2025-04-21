import os

def process_image_folders(input_file, output_file, is_train=True):
    
    with open(input_file, 'r') as f:
        folders = [line.strip() for line in f.readlines()]
    
    image_paths = []
    
    for folder in folders:
        
        folder = "./Data/train/" + folder
        image_files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
        image_files.sort()  
        
        if is_train:
            image_files = image_files[::5]
        else:
            pass
        
        for image_file in image_files:
            image_path = os.path.join(folder, image_file)
            image_paths.append(image_path)
    
    with open(output_file, 'w') as f_out:
        for path in image_paths:
            f_out.write(f"{path}\n")

process_image_folders("./Data/best_folds/train_full_fold_4.txt", "./Data/best_folds/train_fold4_newdata.txt", is_train=True)
process_image_folders("./Data/best_folds/valid_full_fold_4.txt", "./Data/best_folds/valid_fold4_newdata.txt", is_train=False)
