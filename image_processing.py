from PIL import Image
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from rembg import remove
import cv2

def prepare_image_for_prediction(img_path, target_size=(224, 224)):
    assert os.path.exists(img_path), f'Image file not found at path: {img_path}'
    
    img = Image.open(img_path)
    img = remove(img)
    # if img.mode == 'RGBA':
    #     img = img.convert('RGB')
        
    green_background = Image.new("RGB", img.size, (38, 150, 62))
    green_background.paste(img, (0, 0), img)
    img = green_background
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    # Expand dimensions (1, 224, 224, 3) because the model expects a batch of images
    img_array = np.expand_dims(img_array, axis=0)
    # Normalization of the image because the model was trained on normalized images
    img_array /= 255.0
    return img_array

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