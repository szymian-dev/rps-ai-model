# Takes a x random images from the dataset and saves them in a new folder
# Checks that the images are not already in the destination folder

import os
import random
import shutil

SOURCE_DIR = './src_data_unet'
DEST_DIR = './src_data_unet_random'

X = 1200
classes = ['rock', 'paper', 'scissors']

os.makedirs(DEST_DIR, exist_ok=True)
assert os.path.exists(SOURCE_DIR)

for c in classes:
    class_dir = os.path.join(SOURCE_DIR, c)
    class_output_dir = os.path.join(DEST_DIR, c)
    os.makedirs(class_output_dir, exist_ok=True)
    
    files = os.listdir(class_dir)
    random.shuffle(files)
    left_to_copy = X // len(classes)
    for filename in files:
        filepath = os.path.join(class_dir, filename)
        dest_filepath = os.path.join(class_output_dir, filename)
        if not os.path.exists(dest_filepath):
            shutil.copy(filepath, dest_filepath)
            left_to_copy -= 1
            if left_to_copy == 0:
                break
    if left_to_copy > 0:
        print(f'Warning: Could not copy {left_to_copy} images from {c} class, not enough images in the source folder or already copied')
    else:
        print(f'Selected {X // len(classes)} random images from {c} class')
     