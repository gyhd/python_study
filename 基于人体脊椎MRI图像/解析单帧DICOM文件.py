# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:38:54 2019

@author: Maibenben
"""

# 1.1首先导入解析DICOM文件所需的库
import pydicom
import pylab

# 1.2加载DICOM文件
ds = pydicom.read_file(r'MRI_T2\曾祥德_MR1803856\2_11\1.DCM')# 在你机器上DICOM文件的位置

# 1.3相关属性的打印
print("打印所有 DICOM TAG名:\n",ds.dir()) # 打印所有 DICOM TAG名
print("打印包含 'pat' 的 DICOM TAG:\n",ds.dir('pat')) # 打印包含 'pat' 的 DICOM TAG
# 属性值从DICOM TAG中找
print("打印 DICOM TAG相应的属性值:\n",ds.PatientName, ds.PatientSex, ds.PatientPosition, ds.PatientWeight) # 打印 DICOM TAG相应的属性值

print("打印出DICOMTAG编码值(Group, Element):\n",ds.data_element('PatientID'))
print("打印出DICOMTAG编码值VR:\n",ds.data_element('PatientID').VR)
print("打印出DICOMTAG编码值value:\n",ds.data_element('PatientID').value)


pixel_bytes = ds.PixelData # 原始二进制文件
pix = ds.pixel_array       # 像素值矩阵
print("打印矩阵维度:\n",pix.shape) # 打印矩阵维度

pylab.imshow(pix, cmap=pylab.cm.bone)
pylab.show() # cmap 表示 colormap,可以是设置成不同值获得不同显示效果,打印dicom图片





