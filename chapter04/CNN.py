import numpy as np
import tensorflow as tf
import time
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
class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[5,5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7*7*64,))
        self.dense1 = tf.keras.layers.Dense(units=2048, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10, activation=tf.nn.relu)
        #self.dropout = tf.keras.layers.Dropout(0.5)
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        #x = self.dropout(x)
        output = tf.nn.softmax(x)
        return output
@tf.function
def train_one_step(optimizer, model, X, y, batch_index, summary_writer):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        tf.print("batch_index =", batch_index, ",", "loss =", loss)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    return loss
if __name__ == '__main__':
    num_epochs = 5
    batch_size = 50
    learning_rate = 0.001
    data_loader = MNISTLoader()
    num_batches = 400 #int(data_loader.num_train_data // batch_size * num_epochs)

    model = CNN()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    summary_writer = tf.summary.create_file_writer('./tensorboard')
    tf.summary.trace_on(graph=True, profiler=True)
    start_time = time.time()
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        loss = train_one_step(optimizer, model, X, y, batch_index, summary_writer)
        with summary_writer.as_default():
            if batch_index % 100 == 0:
                tf.summary.scalar('train loss', loss, step=batch_index)
    with summary_writer.as_default():
        tf.summary.trace_export(name='model_trace', step=0, profiler_outdir='./tensorboard')
    end_time = time.time()
    print(end_time-start_time)