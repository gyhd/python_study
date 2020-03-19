# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 23:40:20 2019

@author: Maibenben
"""


import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

#matplotlib inline

num_px = 64

my_image = "666.jpg"   # 修改你图像的名字

fname = "C:\\Users\\Maibenben\\Desktop\\" + my_image     # 图片位置     
image = np.array(ndimage.imread(fname, flatten=False))    # 读入图片为矩阵

plt.imshow(image)


