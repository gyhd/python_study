# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 19:38:22 2019

@author: Maibenben
"""

"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('E:\img.jpg',0)
cv2.imshow('cat',img)

cv2.waitKey(0)
cv2.destroyAllWindows()



#在图像上面绘制和写字
import cv2
import numpy as np

img = cv2.imread('E:\img.jpg', 0)
#画一条线
#cv2.line(img,(0,0),(500,350),(255,255,255),15)
#画一个矩形
#cv2.rectangle(img,(15,25),(500,350),(0,0,255),15)
#画一个圆
cv2.circle(img,(300,225), 100, (0,255,0), -1)#粗细为-1代表要填充一个圆

cv2.imshow('image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()

"""


import numpy as np
import cv2

img = cv2.imread('E:\img.jpg',cv2.IMREAD_COLOR)

cv2.line(img,(0,0),(200,300),(255,255,255),50)
#cv2.rectangle(img,(500,250),(1000,500),(0,0,255),15)
#cv2.circle(img,(447,63), 63, (0,255,0), -1)

pts = np.array([[100,50],[200,300],[700,200],[500,100]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img, [pts], True, (0,255,255), 3)

#在图片上面写字
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV Tuts!',(100,50), font, 6, (200,255,155), 1, cv2.LINE_AA)
cv2.imshow('image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()





























