# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:18:31 2019

@author: Maibenben
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import pydicom as dicom
import pylab

pathDicom = r"MRI_T2"

# 包括LCX.dcm和RCA.dcm两个DICOM文件
dicomFileList = []

for root,subdir,fileList in os.walk(pathDicom):
    for fileName in fileList:
        if ".dcm" in fileName.lower():
            print(fileName)
            dicomFileList.append(os.path.join(root,fileName))

# 读取第二个文件：RCA.dcm(60帧)
RefDs=dicom.read_file(dicomFileList[1])

print("len(RefDs):",len(RefDs))
print("type(RefDs):",type(RefDs))

RCA_array=RefDs.pixel_array
print("len(RCA_array):",len(RCA_array))


#展示该文件的第30帧
plt.imshow(RefDs.pixel_array[30],cmap=plt.cm.bone)
plt.show()






