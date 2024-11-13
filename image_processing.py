from PIL import Image
import os
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

def prepare_image_for_prediction(img_path, target_size=(224, 224), grayscale=True):
    assert os.path.exists(img_path), f'Image file not found at path: {img_path}'
    img = Image.open(img_path)
    
    if img.height > img.width:
        img = img.rotate(90, expand=True)
    
    if grayscale:
        img = img.convert('L')
    
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = image.img_to_array(img)
    
    if grayscale:
        img_array = np.expand_dims(img_array, axis=-1)
    
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Only for RGB images
def random_color_and_grayscale_augmentation(image):
    if np.random.rand() < 0.3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        return gray_image
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hue_shift = np.random.uniform(-10, 10) 
        saturation_scale = np.random.uniform(0.8, 1.2) 
        value_scale = np.random.uniform(0.8, 1.2)

        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift, 0, 179)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value_scale, 0, 255)

        augmented_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        augmented_image = augmented_image.astype(np.float32)
        return augmented_image
    
    
def convert_images_to_grayscale(dataset_dir, splits, classes):
    """
    Convert all images in the dataset to grayscale. (In-place operation)
    Args:
        dataset_dir (str): Main dataset directory (e.g. 'dataset').
        splits (list of str): List of dataset splits (e.g. ['train', 'val', 'test']).
        classes (list of str): List of class names (e.g. ['rock', 'paper', 'scissors']).
    """
    counter = 0
    skipped = 0
    print("Converting images to grayscale...")
    for split in splits:
        for cls in classes:
            class_dir = os.path.join(dataset_dir, split, cls)
            
            if not os.path.exists(class_dir):
                raise Exception(f"Directory not found: {class_dir}")

            for filename in os.listdir(class_dir):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    img_path = os.path.join(class_dir, filename)
  
                    img = Image.open(img_path)
                    
                    if img.mode == 'L':
                        skipped += 1
                        continue
                    
                    gray_img = img.convert('L')
                    
                    gray_img.save(img_path)
                    counter += 1

    print(f"Converted {counter} images to grayscale, skipped {skipped} images.")
    
    
def random_contrast_grayscale(image, min_contrast=0.8, max_contrast=1.2):
    contrast_scale = np.random.uniform(min_contrast, max_contrast)
    mean = np.mean(image)
    image = mean + (image - mean) * contrast_scale
    image = np.clip(image, 0, 255)
    return image