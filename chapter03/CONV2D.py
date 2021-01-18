import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'

image = np.array([[
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 2, 1, 0],
    [0, 0, 2, 2, 0, 1, 0],
    [0, 1, 1, 0, 2, 1, 0],
    [0, 0, 2, 1, 1, 0, 0],
    [0, 2, 1, 1, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]], dtype=np.float32)
image = np.expand_dims(image, axis=-1)
W = np.array([[
    [0, 0, -1],
    [0, 1, 0],
    [-2, 0, 2]
]], dtype=np.float32)
b = np.array([1], dtype=np.float32)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=[3,3],
        kernel_initializer=tf.constant_initializer(W),
        bias_initializer=tf.constant_initializer(b)
    )]
)
output = model(image)
print(tf.squeeze(output))