from PIL import Image
import os

d = '../src_data'
o = './src_data_resized'
os.makedirs(o, exist_ok=True)

classes = ['rock', 'paper', 'scissors']

RESIZE_SHAPE = (128, 128)

for c in classes:
    class_dir = os.path.join(d, c)
    class_output_dir = os.path.join(o, c)
    os.makedirs(class_output_dir, exist_ok=True)
    
    for filename in os.listdir(class_dir):
        filepath = os.path.join(class_dir, filename)
        img = Image.open(filepath)
        img = img.resize(RESIZE_SHAPE, Image.Resampling.LANCZOS)
        img.save(os.path.join(class_output_dir, filename))