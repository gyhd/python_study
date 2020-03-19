# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:56:46 2019

@author: Maibenben
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO) #设定输出日志的模式

#我们的程序代码将放在这里
def cnn_model_fn(features, labels, mode):
    #输入层，-1表示自动计算，这里是图片批次大小，宽高各28，最后1表示颜色单色
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    #1号卷积层，过滤32次，核心区域5x5，激活函数relu
    conv1 = tf.layers.conv2d(
        inputs=input_layer,#接收上面创建的输入层输出的张量
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    #1号池化层，接收1号卷积层输出的张量
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    #2号卷积层
    conv2 = tf.layers.conv2d(
        inputs=pool1,#继续1号池化层的输出
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    
    #2号池化层
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    #对2号池化层的输入变换张量形状
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    
    #密度层
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    
    #丢弃层进行简化
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    #使用密度层作为最终输出，unit可能的分类数量
    logits = tf.layers.dense(inputs=dropout, units=10)
    
    #预测和评价使用的输出数据内容
    predictions = {
      #产生预测，argmax输出第一个轴向的最大数值
      "classes": tf.argmax(input=logits, axis=1),
      #输出可能性
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    #以下是根据mode切换的三个不同的方法，都返回tf.estimator.EstimatorSpec对象
  
    #预测
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    #损失函数(训练与评价使用)，稀疏柔性最大值交叉熵
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    #训练，使用梯度下降优化器，
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    #评价函数（上面两个mode之外else）添加评价度量(for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



dir_path = os.path.dirname(os.path.realpath(__file__))
data_path=os.path.join(dir_path,'MNIST_data')
def main(args):
  #载入训练和测试数据
    mnist = input_data.read_data_sets(data_path)
    train_data = mnist.train.images #得到np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images #得到np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    
    #创建估算器
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
    
    #设置输出预测的日志
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    
    #训练喂食函数
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    
    #启动训练
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=[logging_hook])
    
    #评价喂食函数
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    
    #启动评价并输出结果
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)    


#这个文件能够直接运行，也可以作为模块被其他文件载入
if __name__== "__main__":
    tf.app.run()








