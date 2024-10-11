from PIL import Image

def prepare_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    # Expand dimensions (1, 224, 224, 3) because the model expects a batch of images
    img_array = np.expand_dims(img_array, axis=0)
    # Normalization of the image because the model was trained on normalized images
    img_array /= 255.0
    return img_array