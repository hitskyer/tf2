import numpy as np
import tensorflow as tf
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'

num_epochs = 10
batch_size = 32
learning_rate = 0.001
data_dir = './cats_vs_dogs'
train_file = data_dir + '/train/train.tfrecords'
test_file = data_dir + '/test/test.tfrecords'

def _decode_and_resize(example_string):
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    image_decoded = tf.image.decode_jpeg(feature_dict['image'])
    image_resized = tf.image.resize(image_decoded, [256, 256])/255.0
    label = feature_dict['label']
    return image_resized, label
feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}
if __name__ == '__main__':
    train_dataset = tf.data.TFRecordDataset(train_file)
    train_dataset = train_dataset.map(
        map_func=_decode_and_resize,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_dataset = train_dataset.shuffle(buffer_size=23000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 5, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
    model.fit(train_dataset, epochs=num_epochs)

    test_dataset = tf.data.TFRecordDataset(test_file)
    test_dataset = test_dataset.map(
        map_func=_decode_and_resize,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    test_dataset = test_dataset.batch(batch_size)

    print(model.metrics_names)
    print(model.evaluate(test_dataset))