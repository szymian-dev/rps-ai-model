import tensorflow as tf
import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

def load_images_and_masks(image_dir, mask_dir, img_size=(224, 224)):
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
    mask_paths = [os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)]
    
    images = []
    masks = []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = load_img(img_path, target_size=img_size)
        img = img_to_array(img) / 255.0
        
        mask = load_img(mask_path, target_size=img_size, color_mode="grayscale")
        mask = img_to_array(mask) / 255.0
        
        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

def build_unet(input_size=(256, 256, 3)):  # Default input size (256x256x3)
    input_img = Input(shape=input_size)
    # First block (Contracting path)
    enc_block1_conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    enc_block1_conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(enc_block1_conv1)

    # Second block
    enc_block2_pool = MaxPooling2D()(enc_block1_conv2)
    enc_block2_conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(enc_block2_pool)
    enc_block2_conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(enc_block2_conv1)

    # Third block
    enc_block3_pool = MaxPooling2D()(enc_block2_conv2)
    enc_block3_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(enc_block3_pool)
    enc_block3_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(enc_block3_conv1)

    # Fourth block
    enc_block4_pool = MaxPooling2D()(enc_block3_conv2)
    enc_block4_conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(enc_block4_pool)
    enc_block4_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(enc_block4_conv1)

    # Fifth block
    enc_block5_pool = MaxPooling2D()(enc_block4_conv2)
    enc_block5_conv1 = Conv2D(256, (3, 3), activation='relu', padding='same')(enc_block5_pool)
    enc_block5_conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')(enc_block5_conv1)

    # First block in the expansive path (Upsampling)
    dec_block4_up = UpSampling2D((2, 2))(enc_block5_conv2)
    dec_block4_concat = concatenate([enc_block4_conv2, dec_block4_up])
    dec_block4_conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(dec_block4_concat)
    dec_block4_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(dec_block4_conv1)

    # Second block in the expansive path
    dec_block3_up = UpSampling2D((2, 2))(dec_block4_conv2)
    dec_block3_concat = concatenate([enc_block3_conv2, dec_block3_up])
    dec_block3_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(dec_block3_concat)
    dec_block3_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(dec_block3_conv1)

    # Third block in the expansive path
    dec_block2_up = UpSampling2D((2, 2))(dec_block3_conv2)
    dec_block2_concat = concatenate([enc_block2_conv2, dec_block2_up])
    dec_block2_conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(dec_block2_concat)
    dec_block2_conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(dec_block2_conv1)

    # Fourth block in the expansive path
    dec_block1_up = UpSampling2D((2, 2))(dec_block2_conv2)
    dec_block1_concat = concatenate([enc_block1_conv2, dec_block1_up])
    dec_block1_conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(dec_block1_concat)
    dec_block1_conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(dec_block1_conv1)

    # Output layer
    output = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(dec_block1_conv2)

    # Compile the model
    model = Model(inputs=input_img, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

image_dir = "hgr1/original_images"  
mask_dir = "hgr1/skin_masks"  

X, y = load_images_and_masks(image_dir, mask_dir, img_size=(224, 224))

unet_model = build_unet(input_size=(224, 224, 3))
unet_model.summary()

history = unet_model.fit(X, y, batch_size=16, epochs=20, validation_split=0.2)

unet_model.save("unet_model.keras")

# Plot training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
