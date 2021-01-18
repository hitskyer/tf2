import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'

x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:
    y = tf.square(x)
y_grad = tape.gradient(y, x)
print(x)
print(y)
print(y_grad)