
import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import Image

mp_hands = mp.solutions.hands

# Must be in RGB format
def process_image(image, hands_engine):
    assert isinstance(image, Image.Image), "Input image must be a PIL Image object"
    image = np.array(image)
    results = hands_engine.process(image)
    
    if not results.multi_hand_landmarks:
        return None
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for hand_landmarks in results.multi_hand_landmarks:
        points = [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in hand_landmarks.landmark]
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    
    return mask

def generate_masks(image_dir, output_mask_dir, hands_engine, input_shape=(224, 224), output_shape=(64, 64)):
    os.makedirs(output_mask_dir, exist_ok=True)
    total_processed = 0
    total_failed_detection = 0
    
    for filename in os.listdir(image_dir):
        filepath = os.path.join(image_dir, filename)
        image = Image.open(filepath)
        assert image is not None, f"Could not read image {filepath}"
        
        image = image.resize(input_shape, Image.Resampling.LANCZOS)
        
        mask = process_image(image, hands_engine)
        
        total_processed += 1
        if mask is None:
            total_failed_detection += 1
            continue
        
        mask_filename = os.path.join(output_mask_dir, filename)
        im = Image.fromarray(mask)
        im = im.resize(output_shape, Image.Resampling.LANCZOS)
        if im.mode != 'L':
            im = im.convert('L')
        if im.height > im.width:
            im = im.rotate(90, expand=True)
        im.save(mask_filename)
        
    return total_processed, total_failed_detection
    
def create_rps_mediapipe_dataset(input_dir, output_dir, classes, min_detection_confidence=0.5, input_shape=(224, 224), output_shape=(64, 64)):
    hands_engine = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=min_detection_confidence)
    os.makedirs(output_dir, exist_ok=True)
    
    total_processed = 0
    total_failed_detection = 0
    
    for class_name in classes:
        image_dir = os.path.join(input_dir, class_name)
        output_mask_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_mask_dir, exist_ok=True)
        assert os.path.exists(image_dir), f"Directory {image_dir} does not exist"
        
        tp, tf = generate_masks(image_dir, output_mask_dir, hands_engine, input_shape, output_shape)
        total_processed += tp
        total_failed_detection += tf
    
    print(f"Total processed: {total_processed}, total failed detection: {total_failed_detection}")
    print(f"Total failed detection rate: {total_failed_detection / total_processed * 100:.2f}%")