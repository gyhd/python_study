# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:45:21 2019

@author: Maibenben
"""

import tensorflow as tf

# a ‘bottleneck’ building block
def bottleneck(input, channel_out, stride, scope):
    channel_in = input.get_shape()[-1]
    channel = channel_out / 4
    with tf.variable_scope(scope):
        first_layer = tf.layers.conv2d(input, filters=channel, kernel_size=1, strides=stride,
                                       padding='SAME', activation=tf.nn.relu, name='conv1_1x1')
        second_layer = tf.layers.conv2d(first_layer, filters=channel, kernel_size=3, strides=1,
                                        padding='SAME', activation=tf.nn.relu, name='conv2_3x3')
        third_layer = tf.layers.conv2d(second_layer, filters=channel_out, kernel_size=1, strides=1,
                                       padding='SAME', name='conv3_1x1')
        if channel_in != channel_out:
            # projection (option B)
            shortcut = tf.layers.conv2d(input, filters=channel_out, kernel_size=1,
                                        strides=stride, name='projection')
        else:
            shortcut = input   # identify
        output = tf.nn.relu(shortcut + third_layer)
        return output

# 每一个大卷积层的残差块
def residual_block(input, channel_out, stride, n_bottleneck, down_sampling, scope):
    with tf.variable_scope(scope):
        if down_sampling:
            out = bottleneck(input, channel_out, stride=2, scope='bottleneck_1')
        else:
            out = bottleneck(input, channel_out, stride, scope='bottleneck_1')
        for i in range(1, n_bottleneck):
            out = bottleneck(out, channel_out, stride, scope='bottleneck_%i' % (i+1))
        return out
        
# layer_50残差网络架构
def ResNet_50(images):
    with tf.variable_scope('Layer_50'):
        # conv_1
        conv_1 = tf.layers.conv2d(images, filters=64, kernel_size=7, strides=2, padding='SAME',
                                  activation=tf.nn.relu, name='conv1')
        # conv2_x
        max_pooling = tf.layers.max_pooling2d(conv_1, pool_size=3, strides=2,
                                              padding='SAME', name='max_pooling')
        conv_2 = residual_block(max_pooling, 256, stride=1, n_bottleneck=3, down_sampling=False, scope='conv2')
        # conv3_x
        conv_3 = residual_block(conv_2, 512, stride=1, n_bottleneck=4, down_sampling=True, scope='conv_3')
        # conv4_x
        conv_4 = residual_block(conv_3, 1024, stride=1, n_bottleneck=6, down_sampling=True, scope='conv4')
        # conv5_x
        conv_5 = residual_block(conv_4, 2048, stride=1, n_bottleneck=3, down_sampling=True, scope='conv5')
        average_pooling = tf.layers.average_pooling2d(conv_5, pool_size=7, strides=1, name='avg_pooling')
        full_connection = tf.layers.flatten(average_pooling)
        logits = tf.nn.softmax(tf.layers.dense(full_connection, 1000, name='full_connection'))
        return logits





