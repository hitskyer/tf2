import numpy as np
import tensorflow as tf
import argparse
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
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        output = tf.nn.softmax(x)
        return output
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mode', default='test', help='train or test')
parser.add_argument('--num_epochs', default=5)
parser.add_argument('--batch_size', default=50)
parser.add_argument('--learning_rate', default=0.001)
args = parser.parse_args()

data_loader = MNISTLoader()
def train():
    model = MLP()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    num_batches = int(data_loader.num_train_data//args.batch_size * args.num_epochs)
    checkpoint = tf.train.Checkpoint(model=model)
    for batch_index in range(1, num_batches+1):
        X, y = data_loader.get_batch(args.batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            print("batch %d : loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        if batch_index%100 == 0:
            path = checkpoint.save('./save/model.ckpt')
            print("model saved to %s" % (path))
def test():
    model_to_be_restored = MLP()
    checkpoint = tf.train.Checkpoint(model=model_to_be_restored)
    checkpoint.restore(tf.train.latest_checkpoint("./save"))
    y_pred = np.argmax(model_to_be_restored.predict(data_loader.test_data), axis=-1)
    print("test accuracy: %f" % (sum(y_pred == data_loader.test_label)/data_loader.num_test_data))
if __name__ == "__main__":
    if args.mode == 'train':
        train()
    if args.mode == 'test':
        test()