# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 21:15:47 2019

@author: Maibenben
"""


import cv2
import numpy as np



#读取图片
img = cv2.imread('E:\img.jpg', -1)

#降低分辨率，也可以不降低
#img = cv2.pyrDown(img)

#对图像进行二值化操作
ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 64, 127, cv2.THRESH_BINARY)

#检测轮廓，
#输入的三个参数分别为：输入图像、层次类型、轮廓逼近方法
#因为这个函数会修改输入图像，所以上面的步骤使用copy函数将原图像做一份拷贝，再处理
#返回的三个返回值分别为：修改后的图像、图轮廓、层次
image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    #边界框
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #最小矩形区域
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

    #最小闭圆
    (x, y), radius = cv2.minEnclosingCircle(c)
    center = (int(x), int(y))
    radius = int(radius)
    img = cv2.circle(img, center, radius, (255, 0, 0), 2)

cv2.imshow('image', image)
cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
cv2.imshow("contours", img)

cv2.waitKey(0)
cv2.destroyAllWindows()










