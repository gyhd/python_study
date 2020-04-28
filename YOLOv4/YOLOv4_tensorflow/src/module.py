# coding:utf-8
# 网络基本模块

import tensorflow as tf
from src.Activation import activation as act
from src import Activation
slim = tf.contrib.slim

# conv2d
def conv(inputs, out_channels, kernel_size=3, stride=1):
    '''
    inputs:输入tensor
    out_channels:输出的维度
    kernel_size:卷积核大小
    stride:步长
    return:tensor
    ...
    普通卷积:
        input : [batch, height, width, channel]
        kernel : [height, width, in_channels, out_channels]
    '''
    # 补偿边角
    if stride > 1:
        inputs = padding_fixed(inputs, kernel_size)
    
    # 这里可以自定义激活方式, 默认 relu, 可以实现空洞卷积:rate 参数
    inputs = slim.conv2d(inputs, out_channels, kernel_size, stride=stride, 
                                                padding=('SAME' if stride == 1 else 'VALID'))  
    return inputs

# 边缘全零填充补偿卷积缺失
def padding_fixed(inputs, kernel_size):
    '''
    对tensor的周围进行全0填充
    '''
    pad_total = kernel_size - 1
    pad_start = pad_total // 2
    pad_end = pad_total - pad_start
    inputs = tf.pad(inputs, [[0,0], [pad_start, pad_end], [pad_start, pad_end], [0,0]])
    return inputs

# yolo 残差模块实现
def yolo_res_block(inputs, in_channels, res_num, double_ch=False):
    '''
    yolo的残差模块实现
    inputs:输入
    res_num:一共 res_num 个 1,3,s(残差) 模块
    '''
    out_channels = in_channels
    if double_ch:
        out_channels = in_channels * 2

    # 3,1,r,1模块儿
    route = conv(inputs, in_channels*2, stride=2)            
    net = conv(route, out_channels, kernel_size=1)     # in_channels
    route = conv(route, out_channels, kernel_size=1)# in_channels
    
    # 1,3,s模块儿
    for _ in range(res_num):
        tmp = net
        net = conv(net, in_channels, kernel_size=1)
        net = conv(net, out_channels)                                  # in_channels
        # 相加:shortcut 层
        net = tmp + net

    # 1,r,1模块儿
    net = conv(net, out_channels, kernel_size=1)       # in_channels
    # 拼接:route 层
    net = tf.concat([net, route], -1)
    net = conv(net, in_channels*2, kernel_size=1)
    
    return net

# 3*3 与 1*1 卷积核交错卷积实现
def yolo_conv_block(net,in_channels, a, b):
    '''
    net:输入
    a:一共 a 个 1*1 与 3*3 交错卷积的模块
    b:一共 b 个 1*1 卷积模块儿
    '''
    for _ in range(a):
        out_channels = in_channels / 2
        net = conv(net, out_channels, kernel_size=1)
        net = conv(net, in_channels)
    
    out_channels = in_channels
    for _ in range(b):
        out_channels = out_channels / 2
        net = conv(net, out_channels, kernel_size=1)

    return net

# 最大池化模块
def yolo_maxpool_block(inputs):
    '''
    yolo的最大池化模块, 即 cfg 中的 SPP 模块
    inputs:[N, 19, 19, 512]
    return:[N, 19, 19, 2048]
    '''
    max_5 = tf.nn.max_pool(inputs, 5, [1,1,1,1], 'SAME')
    max_9 = tf.nn.max_pool(inputs, 9, [1,1,1,1], 'SAME')
    max_13 = tf.nn.max_pool(inputs, 13, [1,1,1,1], 'SAME')
    # 拼接
    inputs = tf.concat([max_13, max_9, max_5, inputs], -1)
    return inputs

# 上采样模块儿
def yolo_upsample_block(inputs, in_channels, route):
    '''
    上采样模块儿, 宽高加倍
    inputs:主干输入
    route:以前的特征
    '''
    shape = tf.shape(inputs)
    out_height, out_width = shape[1]*2, shape[2]*2
    inputs = tf.compat.v1.image.resize_nearest_neighbor(inputs, (out_height, out_width))
    
    route = conv(route, in_channels, kernel_size=1)

    inputs = tf.concat([route, inputs], -1)
    return inputs

# 特征提取
def extraction_feature(inputs, batch_norm_params, weight_decay):
    '''
    inputs:[N, 416, 416, 3]
    后面再把每个网络给分出来
    '''
    # ########## 下采样模块儿 ##########
    with slim.arg_scope([slim.conv2d], 
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            biases_initializer=None,
                            activation_fn=lambda x: Activation.activation_fn(x, act.MISH),
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
        with tf.variable_scope('Downsample'):
            net = conv(inputs, 32)
            # downsample
            # res1
            net = yolo_res_block(net, 32, 1, double_ch=True)    # *2
            # res2
            net = yolo_res_block(net, 64, 2)
            # res8
            net = yolo_res_block(net, 128, 8)
            # 第54层特征
            # [N, 76, 76, 256]
            up_route_54 = net
            # res8
            net = yolo_res_block(net, 256, 8)
            # 第85层特征
            # [N, 38, 38, 512]
            up_route_85 = net
            # res4
            net = yolo_res_block(net, 512, 4)

    # ########## leaky_relu 激活 ##########
    with slim.arg_scope([slim.conv2d], 
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            biases_initializer=None,
                            activation_fn=lambda x: Activation.activation_fn(x, act.LEAKY_RELU, 0.1),
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
        with tf.variable_scope('leaky_relu'):
            net = yolo_conv_block(net, 1024, 1, 1)
            # 池化:SPP
            # [N, 19, 19, 512] => [N, 19, 19, 2048]
            net = yolo_maxpool_block(net)
            net = conv(net, 512, kernel_size=1)
            net = conv(net, 1024)
            net = yolo_conv_block(net, 1024, 0, 1)
            # 第116层特征, 用作最后的特征拼接
            # [N, 19, 19, 512]
            route_3 = net

            net = yolo_conv_block(net, 512, 0, 1)
            net = yolo_upsample_block(net, 256, up_route_85)

            net = yolo_conv_block(net, 512, 2, 1)
            # 第126层特征，用作最后的特征拼接
            # [N, 38, 38, 256]
            route_2 = net

            # [N, 38, 38, 256] => [N, 38, 38, 128]
            net = yolo_conv_block(net, 256, 0, 1)
            net = yolo_upsample_block(net, 128, up_route_54)

            net = yolo_conv_block(net, 256, 2, 1)
            # 第136层特征, 用作最后的特征拼接
            # [N, 76, 76, 128]
            route_1 = net

    return route_1, route_2, route_3


