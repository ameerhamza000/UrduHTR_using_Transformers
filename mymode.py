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


image_size = 74
patch_size = 6


# image_path  = "images/urdu.jpeg"
image_path2 = "images/cat.jpeg"
tf_image_cat = Image.open(image_path2)
# tf_image2 = Image.open(image_path2)
tf_image_cat_1 = tf_image_cat.resize((image_size, image_size))

# im = cv2.imread(image_path2)
# print(type(im))
# print(im.shape)
# print(type(im.shape))

# channels = image_path2.split()
# num_channels = len(channels)
# print("Number of channels:", num_channels)


plt.figure(figsize=(4, 4))
image = tf_image_cat_1
plt.imshow(image)
plt.axis("off")
print(image.size)

# resized_image = tf.image.resize(
#     tf.convert_to_tensor([image]), size=(image_size, image_size)
# )
tensor_4d = tf.expand_dims(tf_image_cat_1, axis=0)
patches = Patches(patch_size)(tensor_4d)
print(type(patches))
print(tensor_4d.shape)

print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(18, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(27, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")


# patch_block = patches[0][10]
# patch_img = tf.reshape(patch_block, (patch_size, patch_size, 3))
# plt.imshow(patch_img.numpy().astype("uint8"))
