import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

num_epochs = 5
batch_size_per_replica = 64
learing_rate = 0.001
print('strategy init')
strategy = tf.distribute.MirroredStrategy(devices=['/gpu:1', '/gpu:2', '/gpu:3'])
print('Number of devices: %d' % (strategy.num_replicas_in_sync))
batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

def resize(image, label):
    image = tf.image.resize(image, [224, 224])/255.0
    return image, label
dataset = tfds.load('cats_vs_dogs', split=tfds.Split.TRAIN, as_supervised=True)
dataset = dataset.map(map_func=resize).cache()
dataset = dataset.shuffle(1024)
dataset = dataset.batch(batch_size)
with strategy.scope():
    model = tf.keras.applications.MobileNetV2()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learing_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
print('model fit')
start_time = time.time()
model.fit(dataset, epochs=num_epochs)
end_time = time.time()
print('time : %f' % (end_time-start_time))