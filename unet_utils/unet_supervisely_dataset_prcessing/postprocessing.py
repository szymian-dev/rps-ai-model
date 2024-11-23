import os
import numpy as np
from PIL import Image
import shutil as sh
src_dir = './ds_supervisely'
assert os.path.exists(src_dir)
masks_dir = os.path.join(src_dir, 'masks_machine')
assert os.path.exists(masks_dir)
original_images = os.path.join(src_dir, 'img')
assert os.path.exists(original_images)
masks_human = os.path.join(src_dir, 'masks_human')
annotations = os.path.join(src_dir, 'ann')
if os.path.exists(masks_human):
    sh.rmtree(masks_human)
    print('Removed masks_human')
if os.path.exists(annotations):
    sh.rmtree(annotations)
    print('Removed annotations')


destination_dir = 'processed_masks'
os.makedirs(destination_dir, exist_ok=True)

masks_paths = os.listdir(masks_dir)
for mask_path in masks_paths:
    mask = Image.open(os.path.join(masks_dir, mask_path))
    mask = np.array(mask)
    mask = mask.astype(np.uint8)
    if np.all(mask == 0):
        continue
    mask[mask > 0] = 255
    mask = Image.fromarray(mask)
    mask.save(os.path.join(destination_dir, mask_path))
