# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:07:49 2019

@author: Maibenben
"""

import cv2
import numpy as np
import h5py


f = h5py.File(r'C:\Users\Maibenben\Desktop\Python学习\人脸识别\h5\face.model.h5','r')   #打开h5文件
f.keys()   

print(f.keys())#可以查看所有的主键

a = f['model_weights'][:] #取出主键为data的所有的键值
#f.close()
print(a)

