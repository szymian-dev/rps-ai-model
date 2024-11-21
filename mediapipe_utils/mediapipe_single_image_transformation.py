import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import Image

mp_hands = mp.solutions.hands
hands_engine = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.4)

def process_image(image_path, input_shape=(224, 224), output_shape=(64, 64)):
    assert os.path.exists(image_path), f"Image file not found: {image_path}"
    image = Image.open(image_path)
    if image.mode != 'RGB':
        raise ValueError("Input image must be in RGB format")
    if image.height > image.width:
        image = image.rotate(90, expand=True)
    
    image = image.resize(input_shape, Image.Resampling.LANCZOS)
    image = np.array(image)
    results = hands_engine.process(image)
    if not results.multi_hand_landmarks:
        return None
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for hand_landmarks in results.multi_hand_landmarks:
        points = [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in hand_landmarks.landmark]
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    im = Image.fromarray(mask)
    im = im.resize(output_shape, Image.Resampling.LANCZOS)
    if im.mode != 'L':
        im = im.convert('L')
    if im.height > im.width:
        im = im.rotate(90, expand=True)
        
    im_arr = np.array(im)
    im_arr = np.expand_dims(im_arr, axis=-1)
    im_arr = np.expand_dims(im_arr, axis=0)
    im_arr = im_arr / 255.0
    return im_arr
    
    