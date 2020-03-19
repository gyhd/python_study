# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 19:34:47 2019

@author: Maibenben
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def knn_tensorflow():
    """ tensorflow实现knn算法，对mnist数据识别分类
    :return None
    """
    mnist = input_data.read_data_sets("./data/mnist/", one_hot=True)

    # 数据全部取出, 普通pc计算需要半小时左右，如果嫌太慢，可以少取一些数据。
    train_x, train_y = mnist.train.next_batch(60000)
    test_x, test_y = mnist.test.next_batch(100)

    # 占位符
    train_x_p = tf.placeholder(tf.float32, [None, 784])
    test_x_p = tf.placeholder(tf.float32, [784])

    # L1距离计算：dist = sum(|X1-X2|)
    #dist_l1 = tf.reduce_sum(tf.abs(train_x_p + tf.negative(test_x_p)), reduction_indices=1)

    # L2距离计算：dist = sqrt(sum(|X1-X2|^2))
    dist_l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(train_x_p + tf.negative(test_x_p))), reduction_indices=1))

    # 获得最小距离的索引
    prediction = tf.arg_min(dist_l2, 0)

    # 定义准确率
    accuracy = 0.

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)

        for i in range(len(test_x)):
            # 获取最近邻的值得索引
            nn_index = sess.run(prediction, feed_dict={train_x_p: train_x, test_x_p: test_x[i, :]})
            print("测试集第 %d 条,实际值：%d,预测值：%d" % (i, np.argmax(test_y[i]), np.argmax(train_y[nn_index])))

            # 当预测值==真实值时，计算准确率。
            if np.argmax(test_y[i]) == np.argmax(train_y[nn_index]):
                accuracy += 1. / len(test_x)

        print("准确率：%f " % accuracy)

    return None

if __name__ == '__main__':
    knn_tensorflow()









