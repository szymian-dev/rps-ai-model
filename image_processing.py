from PIL import Image
import os
from tensorflow.keras.preprocessing import image
import numpy as np

def prepare_image(img_path, target_size=(224, 224)):
    assert os.path.exists(img_path), f'Image file not found at path: {img_path}'
    
    img = Image.open(img_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    # Expand dimensions (1, 224, 224, 3) because the model expects a batch of images
    img_array = np.expand_dims(img_array, axis=0)
    # Normalization of the image because the model was trained on normalized images
    img_array /= 255.0
    return img_array