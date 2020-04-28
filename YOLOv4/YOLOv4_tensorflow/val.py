# coding:utf-8
# 结果测试

# 解决cudnn 初始化失败的东西: 使用GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
import config
from utils import tools
from src.YOLO import YOLO
import cv2
import numpy as np
from src import Log
import os
from os import path
import time



# 读取图片
def read_img(img_name, width, height):
    '''
    读取一张图片并转化为网络输入格式
    return:网络输入图片, 原始 BGR 图片
    '''
    img_ori = tools.read_img(img_name)
    if img_ori is None:
        return None, None
    img = cv2.resize(img_ori, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.0
    # [416, 416, 3] => [1, 416, 416, 3]
    img = np.expand_dims(img, 0)
    return img, img_ori

# 保存图片
def save_img(img, name):
    '''
    img:需要保存的 mat 图片
    name:保存的图片名
    '''
    if not path.isdir(config.save_dir):
        Log.add_log("message:创建文件夹'"+str(config.save_dir)+"'")
        os.mkdir(config.save_dir)
    img_name = path.join(config.save_dir, name)
    cv2.imwrite(img_name, img)
    return 


def main():
    yolo = YOLO(config.class_num, config.anchors)

    inputs = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, None, None, 3])
    feature_y1, feature_y2, feature_y3 = yolo.forward(inputs, isTrain=False)
    pre_boxes, pre_score, pre_label = yolo.get_predict_result(feature_y1, feature_y2, feature_y3, config.class_num, 
                                                                                                score_thresh=config.score_thresh, iou_thresh=config.iou_thresh, max_box=config.max_box)

    # 初始化
    init = tf.compat.v1.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        ckpt = tf.compat.v1.train.get_checkpoint_state(config.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            Log.add_log("message:存在 ckpt 模型:'"+str(ckpt.model_checkpoint_path)+"'")
        else:
            Log.add_log("message:不存在 ckpt 模型")
            # exit(1)

        # 名字字典
        word_dict = tools.get_word_dict(config.name_file)
        # 色表
        color_table = tools.get_color_table(config.class_num)

        width = config.width
        height = config.height
        
        for name in os.listdir(config.val_dir):
            img_name = path.join(config.val_dir, name)
            if not path.isfile(img_name):
                print("'%s'不是图片" %img_name)
                continue

            start = time.perf_counter()

            img, img_ori = read_img(img_name, width, height)
            if img is None:
                Log.add_log("message:'"+str(img)+"'图片读取错误")
            boxes, score, label = sess.run([pre_boxes, pre_score, pre_label], feed_dict={inputs:img})
            
            end = time.perf_counter()
            print("%s\t, time:%f s" %(img_name, end-start))

            img_ori = tools.draw_img(img_ori, boxes, score, label, word_dict, color_table)

            cv2.imshow('img', img_ori)
            cv2.waitKey(0)

            if config.save_img:
                save_img(img_ori, name)

if __name__ == "__main__":
    Log.add_log("message:进入 val.main() 函数")
    main()
