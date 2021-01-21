import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        self.train_data = np.expand_dims(self.train_data.astype(np.float32)/255.0, axis=-1)
        self.test_data = np.expand_dims(self.test_data.astype(np.float32)/255.0, axis=-1)
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]
    def get_batch(self, batch_size):
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index, :], self.train_label[index]
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
    @tf.function(input_signature=[tf.TensorSpec([None, 28, 28, 1], tf.float32)])
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
num_epochs = 5
batch_size = 50
learning_rate = 0.001
data_loader = MNISTLoader()
model = MLP()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.metrics.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)
log_dir = './tensorboard/SubModel/'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(
    x=data_loader.train_data,
    y=data_loader.train_label,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(data_loader.test_data, data_loader.test_label),
    callbacks=[tensorboard_callback]
)
tf.saved_model.save(model, 'save/SubModel/2', signatures={"serving_default": model.call})
'''
writer = tf.summary.create_file_writer(log_dir)
tf.summary.trace_on(graph=True, profiler=True)
y_pred = model.call(data_loader.test_data[0:1])
with writer.as_default():
    tf.summary.trace_export(
        name="SubModel",
        step=0,
        profiler_outdir=log_dir
    )
'''