# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:10:25 2019

@author: Maibenben
"""


import os
import pydicom
import numpy
from matplotlib import pyplot


# 用lstFilesDCM作为存放DICOM files的列表
PathDicom = r"MRI_t2"  # 与python文件同一个目录下的文件夹
lstFilesDCM = []

# 将所有dicom文件读入
for diName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # 判断文件是否为dicom文件
            print(filename)
            lstFilesDCM.append(os.path.join(diName, filename))  # 加入到列表中

## 将第一张图片作为参考图
RefDs = pydicom.read_file(lstFilesDCM[10])  # 读取第一张dicom图片

# print(RefDs)
# print(RefDs.pixel_array)
# print(RefDs.PatientPosition)
pyplot.imshow(RefDs.pixel_array, cmap=pyplot.cm.bone)
pyplot.show()

# 建立三维数组,分别记录长、宽、层数(也就是dicom数据个数)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
print(ConstPixelDims)

# 得到spacing值 (mm为单位)
# PixelSpacing - 每个像素点实际的长度与宽度,单位(mm)
# SliceThickness - 每层切片的厚度,单位(mm)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

# 三维数据
x = numpy.arange(0.0, (ConstPixelDims[0] + 1) * ConstPixelSpacing[0], ConstPixelSpacing[0])  # 0到（第一个维数加一*像素间的间隔），步长为constpixelSpacing
y = numpy.arange(0.0, (ConstPixelDims[1] + 1) * ConstPixelSpacing[1], ConstPixelSpacing[1])  #
z = numpy.arange(0.0, (ConstPixelDims[2] + 1) * ConstPixelSpacing[2], ConstPixelSpacing[2])  #
print(len(x),"xxxx")

ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

# 遍历所有的dicom文件，读取图像数据，存放在numpy数组中
for filenameDCM in lstFilesDCM:
    ds = pydicom.read_file(filenameDCM)
    ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

# 轴状面显示
# dpi是指每英寸的像素数,dpi越大,表示打印出来的图片越清晰。不是指图片的大小.
# 像素用在显示领域 分辨率用在打印领域 也就是你的图像是用来打印的时候才去考虑分辨率的问题
pyplot.figure(dpi=1000)

# 将坐标轴都变为同等长度
# pyplot.axes().set_aspect('equal', 'datalim')
pyplot.axes().set_aspect('equal')

# 将图片变为gray颜色
pyplot.set_cmap(pyplot.gray())

pyplot.imshow(ArrayDicom[:, :, 360])# 第三个维度表示现在展示的是第几层 
pyplot.show() 

# 冠状面显示 
pyplot.figure(dpi=100) 
pyplot.axes().set_aspect('equal', 'datalim') 
pyplot.set_cmap(pyplot.gray()) 
pyplot.imshow(ArrayDicom[:, 90, :])
pyplot.show()









