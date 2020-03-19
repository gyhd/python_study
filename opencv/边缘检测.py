# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 21:09:24 2019

@author: Maibenben
"""

import cv2

img = cv2.imread('E:\img.jpg', flags=cv2.IMREAD_GRAYSCALE)
GBlur = cv2.GaussianBlur(img, (3, 3), 0)
canny = cv2.Canny(GBlur, 50, 150)

cv2.imshow('img', img)
cv2.imshow('canny', canny)

cv2.waitKey(0)
cv2.destroyAllWindows()





