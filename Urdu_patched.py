import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import cv2
import numpy as np

def convert_to_3d(image_path):
    # Open the 2D image using PIL
    image_2d = Image.open(image_path)
    # Get the dimensions of the 2D image
    width, height = image_2d.size
    # Create a new 3D image with RGB mode
    image_3d = Image.new("RGB", (width, height))
    # Copy the 2D image to all three channels of the 3D image
    for x in range(width):
        for y in range(height):
            pixel_value = image_2d.getpixel((x, y))
            image_3d.putpixel((x, y), (pixel_value, pixel_value, pixel_value))
    return image_3d

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    
image_size = 72
patch_size = 12


input_image_path = r"images/urdu.jpeg"
img_3d = convert_to_3d(input_image_path)
#Resize the Input image
tf_image = img_3d.resize((797, image_size))

#plot the resized image
plt.figure(figsize=(4, 4))
image = tf_image
plt.imshow(image, cmap='YlGnBu')
plt.axis("off")
print(image.size)

tensor_4d = tf.expand_dims(tf_image, axis=0)
patches = Patches(patch_size)(tensor_4d)
print((patches.shape))
print(tensor_4d.shape)

print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")



n = int(np.sqrt(patches.shape[1]))  #number of rows and col in one patch.
plt.figure()
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")





