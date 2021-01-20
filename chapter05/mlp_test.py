import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        self.train_data = np.expand_dims(self.train_data.astype(np.float32)/255.0, axis=-1)
        self.test_data = np.expand_dims(self.test_data.astype(np.float32)/255.0, axis=-1)
        #self.test_data = self.test_data.astype(np.float32)/255.0
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]
    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]
batch_size = 50
model = tf.saved_model.load('save/2')
data_loader = MNISTLoader()
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(data_loader.num_test_data//batch_size)
for batch_index in range(num_batches):
    sindex, eindex = batch_index*batch_size, (batch_index+1)*batch_size
    y_pred = model(data_loader.test_data[sindex:eindex])
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[sindex:eindex], y_pred=y_pred)
print('test accuracy: %f' % (sparse_categorical_accuracy.result()))