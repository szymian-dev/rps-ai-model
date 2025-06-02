import numpy as np
import os
from PIL import Image   
import tensorflow as tf
from skimage.morphology import remove_small_holes, remove_small_objects, square, disk, binary_closing


   
unet_model = tf.keras.models.load_model('C:/Adrian/Projects/Bachelor/aimodel/models_segmantation/unet_segmentation_ckp.keras')

def check_if_mask_almost_empty(mask):
    count_non_zero = np.count_nonzero(mask)
    if count_non_zero < 900:
        return True
    
def process_image(image_path, img_size=(128, 128)):
    assert os.path.exists(image_path), f"Image file not found: {image_path}"
    image = Image.open(image_path)
    if image.mode != 'RGB':
        raise ValueError("Input image must be in RGB format")
    if image.height > image.width:
        image = image.rotate(90, expand=True)
    
    image = image.resize(img_size, Image.Resampling.LANCZOS)
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    result = unet_model.predict(image)
    pred_mask = (result.squeeze() > 0.5).astype(np.uint8) 
    mask_bool = pred_mask.astype(bool)
    processed_mask = remove_small_objects(mask_bool, min_size=128)
    processed_mask = remove_small_holes(processed_mask, area_threshold=256).astype(np.uint8)
    
    if check_if_mask_almost_empty(processed_mask):
        return None
    
    assert np.all(np.isin(processed_mask, [0, 1])), 'Mask values should be 0 or 1'
    

    im_arr = np.expand_dims(processed_mask, axis=-1)
    im_arr = np.expand_dims(im_arr, axis=0)
    return im_arr