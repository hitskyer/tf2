import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'

A = tf.constant([[1, 2],[3, 4]])
B = tf.constant([[5, 6],[7, 8]])
C = tf.matmul(A, B)

print(C)