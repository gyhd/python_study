# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 21:05:54 2019

@author: Maibenben
"""

import cv2
import numpy as np
from scipy import ndimage


kernel_3x3 = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1],
])

kernel_5x5 = np.array([
    [-1, -1, -1, -1, -1],
    [-1, -1,  2, -1, -1],
    [-1,  2,  4,  2, -1],
    [-1, -1,  2, -1, -1],
    [-1, -1, -1, -1, -1],
])

img = cv2.imread('C:\Users\Maibenben\Desktop\product\hegepin\2019_01_10__14_20_36_464_A.jpg', flags=cv2.IMREAD_GRAYSCALE)

k3 = ndimage.convolve(img, kernel_3x3)
k5 = ndimage.convolve(img, kernel_5x5)

GBlur = cv2.GaussianBlur(img, (11, 11), 0)
g_hpf = img - GBlur



cv2.imshow('img', img)
cv2.imshow('3x3', k3)
cv2.imshow('5x5', k5)
cv2.imshow('g_hpf', g_hpf)
cv2.waitKey()
cv2.destroyAllWindows()





