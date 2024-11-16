import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands

def process_image(image, hands):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for hand_landmarks in results.multi_hand_landmarks:
        points = [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in hand_landmarks.landmark]
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    
    return mask

def generate_masks(image_dir, output_mask_dir):
    os.makedirs(output_mask_dir, exist_ok=True)
    
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    for filename in os.listdir(image_dir):
        filepath = os.path.join(image_dir, filename)
        image = cv2.imread(filepath)
        if image is None:
            continue
        
        mask = process_image(image, hands)
        
        if mask is None:
            continue
        
        mask_filename = os.path.join(output_mask_dir, filename)
        cv2.imwrite(mask_filename, mask)

generate_masks("predict_images", "predict_masks")