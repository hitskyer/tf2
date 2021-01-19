import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'

if __name__ == '__main__':
    fig, axs = plt.subplots(1,2)
    image_string = tf.io.read_file('cats_vs_dogs/train/cats/cat.5207.jpg')
    image_decoded = tf.image.decode_jpeg(image_string)
    axs[0].set_title('raw')
    axs[0].imshow(image_decoded.numpy())
    image_resized = tf.image.resize(image_decoded, [256, 256])
    axs[1].set_title('resize')
    axs[1].imshow(np.asarray(image_resized.numpy(), dtype='uint8'))
    plt.show()
    print('done')