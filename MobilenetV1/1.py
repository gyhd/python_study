# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:52:16 2019

@author: Maibenben
"""

# 使用 slim 来实现
def _depthwise_separable_conv(inputs,
                              num_pwc_filters,
                              kernel_width,
                              phase,
                              sc,
                              padding='SAME',
                              width_multiplier=1,
                              downsample=False):
    """ Helper function to build the depth-wise separable convolution layer.
    """
    num_pwc_filters = round(num_pwc_filters * width_multiplier)
    _stride = 2 if downsample else 1

    # skip pointwise by setting num_outputs=None
    depthwise_conv = slim.separable_convolution2d(inputs,
                                                  num_outputs=None,
                                                  stride=_stride,
                                                  depth_multiplier=1,
                                                  kernel_size=[kernel_width, kernel_width],
                                                  padding=padding,
                                                  activation_fn=None,
                                                  scope=sc + '/depthwise_conv')

    bn = slim.batch_norm(depthwise_conv, activation_fn=tf.nn.relu, is_training=phase, scope=sc + '/dw_batch_norm')
    pointwise_conv = slim.convolution2d(bn,
                                        num_pwc_filters,
                                        kernel_size=[1, 1],
                                        activation_fn=None,
                                        scope=sc + '/pointwise_conv')
    bn = slim.batch_norm(pointwise_conv, activation_fn=tf.nn.relu, is_training=phase, scope=sc + '/pw_batch_norm')
    return bn
