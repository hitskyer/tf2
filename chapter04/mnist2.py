import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'

def rot90(image, label):
    image = tf.image.rot90(image)
    return image, label
(train_data, train_label), (_, _) = tf.keras.datasets.mnist.load_data()
train_data = np.expand_dims(train_data.astype(np.float32)/255.0, axis=-1)
mnist_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
mnist_dataset = mnist_dataset.map(rot90)

for image, label in mnist_dataset:
    plt.title(label.numpy())
    plt.imshow(image.numpy()[:, :, 0])
    plt.show()