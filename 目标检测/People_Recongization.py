# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:58:29 2019

@author: Maibenben
"""

import cv2

# 确定某矩形是否完全包含在另一个矩形中
def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox+ow < ix+iw and oy + oh < iy + ih

# 绘制矩形来框住检测到的人
def draw_person(image, person):
    x, y, w, h = person
    cv2.rectangle(img, (x, y), (x+w, y + h), (0, 255, 255), 2)


# 导入图像，
img = cv2.imread(r"C:\Users\Maibenben\Desktop\huge2.jpg")
# 实例化HOGDescriptor对象，作为检测人的检测器
hog = cv2.HOGDescriptor()
# 设置线性SVM分类器的系数
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 该和人脸算法不一样，不需要在使用目标检测方法前将原始图像转换为灰度形式
# 该方法返回一个与矩形相关的数组，用户可用该数组在图形上绘制形状
# 若图形上的矩形存在有包含与被包含的关系，说明检测出现了错误
# 被包含的图形应该被丢弃，此过程由is_inside来实现
# 在输入图像中检测不同大小的对象。检测到的对象作为列表返回
found, w = hog.detectMultiScale(img)

found_filtered = []

# 遍历检测结果，丢弃不含有检测目标区域的矩形。
for ri, r in enumerate(found):
    for qi, q in enumerate(found):
        if ri != qi and is_inside(r, q):
            break
        else:
            found_filtered.append(r)

for person in found_filtered:
    draw_person(img, person)

cv2.imshow("people detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


