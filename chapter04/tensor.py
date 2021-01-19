import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'

X = tf.constant([2013, 2014, 2015, 2016,2017])
Y = tf.constant([12000, 14000, 15000, 16500, 17500])
dataset = tf.data.Dataset.from_tensor_slices((X, Y))
for x, y in dataset:
    print(x.numpy(), y.numpy())