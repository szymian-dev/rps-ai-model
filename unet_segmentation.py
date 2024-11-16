import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

model = load_model('unet_model.keras')

def load_test_images(image_dir, img_size=(224, 224)):
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
    images = []
    
    for img_path in image_paths:
        img = load_img(img_path, target_size=img_size)
        img = img_to_array(img) / 255.0
        images.append(img)
    
    return np.array(images)

class_name = 'scissors'
test_image_dir = 'src_data/' + class_name

test_images = load_test_images(test_image_dir)

predictions = model.predict(test_images)

predictions = (predictions > 0.5).astype(np.uint8)

output_dir = 'src_masks/' + class_name

os.makedirs(output_dir, exist_ok=True)

for i, pred in enumerate(predictions):
    pred_mask = np.squeeze(pred, axis=-1)
    output_path = os.path.join(output_dir, f"pred_mask_{i+1}.png")
    cv2.imwrite(output_path, pred_mask * 255)

print("Predictions saved!")
