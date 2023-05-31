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

# dataset_dir = r'C:\Users\Windows10\Documents\Untitled Folder\UHWR\Dataset\images'
# # Get a list of image file names in the dataset directory
# image_files = os.listdir(dataset_dir)
# # Select a random image file from the list
# random_image_file = random.choice(image_files)
# # Load the random image using PIL
# imag = Image.open(os.path.join(dataset_dir, random_image_file))
# print(type(imag))
# tf_image = np.array(imag)
# print(type(tf_image))
image_path  = "C:\\Users\\Windows10\\Documents\\Untitled Folder\\cat.png"
tf_image = Image.open(image_path)
tf_image = tf_image.resize((72,72))

#batch_size = 256
image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


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


plt.figure(figsize=(15, 115))
plt.imshow(image)
#print(tf_image.size)
plt.axis("off")

resized_image = tf.image.resize(
    tf.convert_to_tensor([tf_image]), size=(image_size, image_size)
)
#print(resized_image.shape)
#Transform 3d into 4d tensor
tensor_4d = tf.expand_dims(resized_image, axis=0)


#print(tensor_4d.shape)

patches = Patches(patch_size)(tensor_4d)

print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")


n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(10, 10))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    image1=tf.image.decode_jpeg(patch_img,channels=3)
    plt.imshow(image1)
    plt.axis("off")

    

