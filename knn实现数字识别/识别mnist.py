# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:24:30 2019

@author: Maibenben
"""


import numpy as np
from Image import image2onebit as it
import sys
from tensorflow.examples.tutorials.mnist import input_data
import math
import datetime


#KNN算法主体：计算测试样本与每一个训练样本的距离
def get_index(train_data,test_data, i):
	#1、 np.argmin(np.sqrt(np.sum(np.square(test_data[i]-train_data),axis=1)))
	#2、a数组存入：测试样本与每一个训练样本的距离
	all_dist = np.sqrt(np.sum(np.square(test_data[i]-train_data),axis=1)).tolist()
	return all_dist

#KNN算法主体：计算查找最近的K个训练集所对应的预测值
def get_number(all_dist):
	all_number = []
	min_index = 0
	#print('距离列表：', all_dist,)
	for k in range(Nearest_Neighbor_number):
		# 最小索引值 = 最小距离的下标编号
		min_index = np.argmin(all_dist)
		#依据最小索引值（最小距离的下标编号），映射查找到预测值
		ss = np.argmax((train_label[min_index])).tolist()
		print('第',k+1,'次预测值:',ss)
		#将预测值改为字符串形式存入新元组bb中
		all_number = all_number + list(str(ss))
		#在距离数组中，将最小的距离值删去
		min_number = min(all_dist)
		xx = all_dist.index(min_number)
		del all_dist[xx]
	print('预测值总体结果：',all_number)
	return all_number

#KNN算法主体：在K个预测值中，求众数，找到分属最多的那一类，输出
def get_min_number(all_number):
	c = []
	#将string转化为int，传入新列表c
	for i in range(len(all_number)):
		c.append(int(all_number[i]))
	#求众数
	new_number = np.array(c)
	counts = np.bincount(new_number)
	return np.argmax(counts)

t1 = datetime.datetime.now()      #计时开始
print('说明：训练集数目取值范围在[0,60000],K取值最好<10\n' )
train_sum = int(input('输入训练集数目：'))
Nearest_Neighbor_number = int(input('选取最邻近的K个值，K='))

#依照文件名查找，读取训练与测试用的图片数据集
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
#取出训练集数据、训练集标签
train_data, train_label = mnist.train.next_batch(train_sum)

#调用自创模块内函数read_image()：依照路径传入图片处理，将图片信息转换成numpy.array类型
x1_tmp = it.read_image("png/55.png")
test_data = it.imageToArray(x1_tmp)
test_data = np.array(test_data)
#print('test_data',test_data)
#调用自创模块内函数show_ndarray()：用字符矩阵打印图片
it.show_ndarray(test_data)

#KNN算法主体
all_dist = get_index(train_data,test_data,0)
all_number = get_number(all_dist)
min_number = get_min_number(all_number )
print('最后的预测值为：',min_number)

t2=datetime.datetime.now()
print('耗 时 = ',t2-t1)





