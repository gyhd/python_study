# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:01:38 2019

@author: Maibenben
"""


import cv2

img = cv2.imread('E:\img.jpg', 0)
cv2.imshow('1', img)

cv2.waitKey()
cv2.destroyAllWindows()




