# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:27:54 2019

@author: Maibenben
"""

import tensorflow as tf
from tensorflow.contrib.layers import flatten


def AlexNet(images):
    # 图像规格227*227*3，卷积核大小为11*11，步长为4，深度为96, 输出为55*55*96
    with tf.name_scope('conv_1'):
        conv1_w = tf.Variable(tf.truncated_normal(shape=[11, 11, 3, 96], dtype=tf.float32, mean=0,
                                                  stddev=0.1), name='weights')
        conv1_b = tf.Variable(tf.constant(value=0.0, shape=[96], dtype=tf.float32),
                              trainable=True, name='biases')
        conv_1 = tf.nn.conv2d(images, conv1_w, strides=[1, 4, 4, 1], padding='VALID') + conv1_b
        conv_1 = tf.nn.relu(conv_1, name='conv1')

    # 输入55*55*96，过滤器大小为3*3，步长为2，输出为27*27*96
    with tf.name_scope('lrn1_and_pooling1'):
        lrn_1 = tf.nn.lrn(conv_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='LRN_1')
        pooling_1 = tf.nn.max_pool(lrn_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                   padding='VALID', name='pooling_1')

    # 输入为27*27*96，卷积核大小为5*5，步长为1，填充为2，深度为256，输出为27*27*256
    with tf.name_scope('conv_2'):
        conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 96, 256], dtype=tf.float32, mean=0,
                                                  stddev=0.1), name='weights')
        conv2_b = tf.Variable(tf.constant(value=0.0, shape=[256], dtype=tf.float32),
                              trainable=True, name='biases')
        conv_2 = tf.nn.conv2d(pooling_1, conv2_w, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
        conv_2 = tf.nn.relu(conv_2, name='conv2')

    # 输入为27*27*256，过滤器大小为3*3,步长为2,输出为13*13*256
    with tf.name_scope('lrn2_and_pooling2'):
        lrn_2 = tf.nn.lrn(conv_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn_2')
        pooling_2 = tf.nn.max_pool(lrn_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                   padding='VALID', name='pooling_2')

    # 输入为13*13*256，卷积核大小为3*3，步长为1，填充为1，输出为13*13*384
    with tf.name_scope('conv_3'):
        conv3_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 384], dtype=tf.float32, mean=0,
                                                 stddev=0.1), name='weights')
        conv3_b = tf.Variable(tf.constant(value=0.0, shape=[384], dtype=tf.float32),
                              trainable=True, name='biases')
        conv_3 = tf.nn.conv2d(pooling_2, conv3_w, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
        conv_3 = tf.nn.relu(conv_3, name='conv_3')

    # 输入为13*13*384，卷积核大小为3*3，步长为1，填充为1，输出为13*13*384
    with tf.name_scope('conv_4'):
        conv4_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 384, 384], dtype=tf.float32, mean=0,
                                                  stddev=0.1), name='weights')
        conv4_b = tf.Variable(tf.constant(value=0.0, shape=[384], dtype=tf.float32), name='biases')
        conv_4 = tf.nn.conv2d(conv_3, conv4_w, strides=[1, 1, 1, 1], padding='SAME') + conv4_b
        conv_4 = tf.nn.relu(conv_4, name='conv_4')

    # 输入为13*13*384，卷积核大小为3*3，步长为1，填充为1，输出为13*13*256
    with tf.name_scope('conv_5'):
        conv5_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 384, 256], dtype=tf.float32, mean=0,
                                                  stddev=0.1), name='weights')
        conv5_b = tf.Variable(tf.constant(value=0.0, shape=[256], dtype=tf.float32), name='biases')
        conv_5 = tf.nn.conv2d(conv_4, conv5_w, strides=[1, 1, 1, 1], padding='SAME') + conv5_b
        conv_5 = tf.nn.relu(conv_5, name='conv_5')

    #输入为13*13*256，过滤器大小为3*3，步长为2，输出为6*6*256
    with tf.name_scope('pooling_3'):
        pooling_3 = tf.nn.max_pool(conv_5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                   padding='VALID', name='pooling_3')
        print('pooling_3:', pooling_3.shape)
    
    fc_1 = flatten(pooling_3)
    dim = fc_1.shape[1].value

    with tf.name_scope('full_connection_1'):
        fc1_w = tf.Variable(tf.truncated_normal(shape=[dim, 4096], dtype=tf.float32, mean=0,
                                                stddev=0.1), name='weights')
        fc1_b = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[4096]), name='biases')
        fc1 = tf.matmul(fc_1, fc1_w) + fc1_b

    with tf.name_scope('full_connection_2'):
        fc2_w = tf.Variable(tf.truncated_normal(shape=[4096, 4096], dtype=tf.float32, mean=0,
                                                stddev=0.1), name='weights')
        fc2_b = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[4096]), name='biases')
        fc2 = tf.matmul(fc1, fc2_w) + fc2_b

    with tf.name_scope('full_connection_3'):
        fc3_w = tf.Variable(tf.truncated_normal(shape=[4096, 1000], dtype=tf.float32, mean=0,
                                                stddev=0.1), name='weights')
        fc3_b = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[1000]), name='biases')
        logits = tf.matmul(fc2, fc3_w) + fc3_b
        return logits






