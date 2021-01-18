import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'

X = tf.constant([[1.0, 2.0, 3.0],[2.0, 4.0, 6.0]])
y = tf.constant([[10.0],[20.0]])

class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )
    def call(self, input):
        output = self.dense(input)
        return output
model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(1000):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.reduce_mean(tf.square(y_pred-y))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
print("------------------------------")
print(model.variables)