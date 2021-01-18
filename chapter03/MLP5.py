import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'

class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        self.train_data = np.expand_dims(self.train_data.astype(np.float32)/255.0, axis=-1)
        self.test_data = np.expand_dims(self.test_data.astype(np.float32)/255.0, axis=-1)
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]
    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]
class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation
    def build(self, input_shape):
        self.w = self.add_variable(name='w',
            shape=[input_shape[-1], self.units]
        )
        self.b = self.add_variable(name='b',
            shape=[self.units]
        )
    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        if self.activation:
            y_pred = self.activation(y_pred)
        return y_pred
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = DenseLayer(units=100, activation=tf.nn.relu)
        self.dense2 = DenseLayer(units=10)
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        output = tf.nn.softmax(x)
        return output
num_epochs = 5
batch_size = 50
learning_rate = 0.001
model = MLP()
data_loader = MNISTLoader()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.metrics.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)
model.fit(data_loader.train_data, data_loader.train_label, epochs=num_epochs, batch_size=batch_size)
print(model.evaluate(data_loader.test_data, data_loader.test_label))