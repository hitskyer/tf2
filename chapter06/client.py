import tensorflow as tf
import json
import numpy as np
import requests

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
data_loader = MNISTLoader()
data = json.dumps({
    "instances": data_loader.test_data[0:100].tolist()
})
headers = {"content-type": "application/json"}
json_response = requests.post(
    'http://localhost:8501/v1/models/MLP:predict',
    data=data, headers=headers
)
print(json_response.text)
predictions = np.array(json.loads(json_response.text)['predictions'])
print(np.argmax(predictions, axis=-1))
print(data_loader.test_label[0:100])
print([[i, a, b] for i, (a,b) in enumerate(zip(np.argmax(predictions, axis=-1), data_loader.test_label[0:100])) if a != b])