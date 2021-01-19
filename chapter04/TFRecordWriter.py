import numpy as np
import tensorflow as tf
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'

num_epochs = 10
batch_size = 32
learning_rate = 0.001
data_dir = './cats_vs_dogs'

train_cats_dir = data_dir + '/train/cats/'
train_dogs_dir = data_dir + '/train/dogs/'
train_file = data_dir + '/train/train.tfrecords'

test_cats_dir = data_dir + '/test/cats/'
test_dogs_dir = data_dir + '/test/dogs/'
test_file = data_dir + '/test/test.tfrecords'

if __name__ == '__main__':
    train_cat_files = [train_cats_dir+filename for filename in os.listdir(train_cats_dir)]
    train_dog_files = [train_dogs_dir + filename for filename in os.listdir(train_dogs_dir)]
    train_filenames = train_cat_files + train_dog_files
    train_labels = [0]*len(train_cat_files) + [1]*len(train_dog_files)
    with tf.io.TFRecordWriter(train_file) as writer:
        for filename, label in zip(train_filenames, train_labels):
            image = open(filename, 'rb').read()
            feature = {
                'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    test_cat_files = [test_cats_dir+filename for filename in os.listdir(test_cats_dir)]
    test_dog_files = [test_dogs_dir + filename for filename in os.listdir(test_dogs_dir)]
    test_filenames = test_cat_files + test_dog_files
    test_labels = [0]*len(test_cat_files) + [1]*len(test_dog_files)
    with tf.io.TFRecordWriter(test_file) as writer:
        for filename, label in zip(test_filenames, test_labels):
            image = open(filename, 'rb').read()
            feature = {
                'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())