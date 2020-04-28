# coding: utf-8
# for more details about the yolo darknet weights file, refer to
# https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe

from __future__ import division, print_function

import os
import sys
import tensorflow as tf
import numpy as np
import config
from src.YOLO import YOLO

from utils.misc_utils import load_weights

weight_path = './yolo_weights/yolov4.weights'
save_path = './yolo_weights/yolov4.ckpt'

yolo = YOLO(80, config.anchors)
with tf.Session() as sess:
    inputs = tf.placeholder(tf.float32, [1, 608, 608, 3])

    feature = yolo.forward(inputs, isTrain=False)

    saver = tf.train.Saver(var_list=tf.global_variables())

    load_ops = load_weights(tf.global_variables(), weight_path)
    sess.run(load_ops)
    saver.save(sess, save_path=save_path)
    print('TensorFlow model checkpoint has been saved to {}'.format(save_path))
    

