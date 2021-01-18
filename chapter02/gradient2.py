import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'

X = tf.constant([[1., 2.],[3., 4.]])
y = tf.constant([[1.],[2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = tf.reduce_sum(tf.square(tf.matmul(X, w)+b-y))
w_grad, b_grad = tape.gradient(L, [w,b])
print(L)
print(w_grad)
print(b_grad)