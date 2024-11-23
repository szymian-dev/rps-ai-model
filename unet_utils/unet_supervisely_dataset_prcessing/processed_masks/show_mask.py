import numpy as np
import os
from PIL import Image

images = os.listdir('./')
img = Image.open(images[0])
img = np.array(img)
print(img.shape)
print(img.dtype)
print(img.min(), img.max())
print(np.unique(img))
print(img)