import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'

#定义一个随机数（标量）
random_float = tf.random.uniform(shape=())
#定义一个有2个元素的零向量
zero_vector = tf.zeros(shape=(2))
#定义两个2*2的常量矩阵
A = tf.constant([[1., 2.],[3., 4.]])
B = tf.constant([[5., 6.],[7., 8.]])

print(A.shape)
print(A.dtype)
print(A.numpy())

C = tf.add(A, B)
D = tf.matmul(A, B)

print(C)
print(D)