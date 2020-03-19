# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:26:23 2019

@author: Maibenben
"""

"""
	作者：囚生CY
	平台：CSDN
	时间：2018/10/09
	转载请注明原作者
	创作不易，仅供分享
"""

import random

from tranToXls import *


"""
# 第1组

d1 = 20
d2 = 33
d3 = 46
T1 = 400
T2 = 378
To = 28
Te = 31
Tc = 25

"""
# 第2组

d1 = 23
d2 = 41
d3 = 59
T1 = 280
T2 = 500
To = 30
Te = 35
Tc = 30


# 第3组

"""
d1 = 18
d2 = 32
d3 = 46
T1 = 455
T2 = 182
To = 27
Te = 32
Tc = 25
"""


cncT = [To,Te,To,Te,To,Te,To,Te]

tm = [

	[0,0,d1,d1,d2,d2,d3,d3],

	[0,0,d1,d1,d2,d2,d3,d3],

	[d1,d1,0,0,d1,d1,d2,d2],

	[d1,d1,0,0,d1,d1,d2,d2],

	[d2,d2,d1,d1,0,0,d1,d1],

	[d2,d2,d1,d1,0,0,d1,d1],

	[d3,d3,d2,d2,d1,d1,0,0],

	[d3,d3,d2,d2,d1,d1,0,0],

]

Type = [0,1,0,1,1,1,0,1]												 # CNC刀具分类

 

A = []																	 # 储存第一道工序的CNC编号

B = []																	 # 储存第二道工序的CNC编号

for i in range(len(Type)):

	if Type[i]:

		B.append(i)

	else:

		A.append(i)

 

def init_first_round():													 # 第一圈初始化（默认把所有第一道CNC按顺序加满再回到当前位置全部加满）

	state = [0 for i in range(8)] 									 	 # 记录CNC状态（还剩多少秒结束，0表示空闲）

	isEmpty = [1 for i in range(8)]										 # CNC是否为空

	log = [0 for i in range(8)]											 # 记录每台CNC正在加工第几件物料

	count1 = 0

	rgv = 0																 # rgv状态（0表示空车，1表示载着半成品）

	currP = 0

	total = 0

	seq = []

	flag = False

	for i in range(len(Type)):

		if Type[i]==0:

			seq.append(i)

			flag = True

	currP = seq[0]

	seq.append(currP)

	count1,rgv,currP,total = simulate(seq,state,isEmpty,log,count1,rgv,currP,total)

	return state,isEmpty,log,count1,rgv,currP,total,seq

 

def update(state,t):

	for i in range(len(state)):

		if state[i] < t:

			state[i] = 0

		else:

			state[i] -= t

 

def simulate(seq,state,isEmpty,log,count1,rgv,currP,total,fpath="log.txt"):	# 给定了一个序列模拟它的过程以及返回结果（主要用于模拟并记录）

	index = 0

	temp = 0

	pro1 = {}															 # 第一道工序的上下料开始时间

	pro2 = {}															 # 第二道工序的上下料开始时间

	f = open(fpath,"a")

	while index<len(seq):

		print(isEmpty)

		nextP = seq[index]

		t = tm[currP][nextP]

		total += t

		update(state,t)

		if Type[nextP]==0:												 # 如果下一个位置是第一道工作点

			count1 += 1

			if isEmpty[nextP]:											 # 如果下一个位置是空的

				f.write("第{}个物料的工序一上料开始时间为{}\tCNC编号为{}号\n".format(count1,total,nextP+1))

				t = cncT[nextP]

				total += t

				update(state,t)

				state[nextP] = T1										 # 更新当前的CNC状态

				isEmpty[nextP] = 0										 # 就不空闲了

			else:														 # 如果没有空闲

				if state[nextP] > 0:									 # 如果还在工作就等待结束

					t = state[nextP]

					total += t

					update(state,t)

				f.write("第{}个物料的工序一下料开始时间为{}\tCNC编号为{}号\n".format(log[nextP],total,nextP+1))

				f.write("第{}个物料的工序一上料开始时间为{}\tCNC编号为{}号\n".format(count1,total,nextP+1))

				t = cncT[nextP]											 # 完成一次上下料

				total += t

				update(state,t)

				state[nextP] = T1

				rgv = log[nextP]

			log[nextP] = count1

		else:															 # 如果下一个位置是第二道工作点

			if isEmpty[nextP]:											 # 如果下一个位置是空的

				f.write("第{}个物料的工序二上料开始时间为{}\tCNC编号为{}号\n".format(rgv,total,nextP+1))

				t = cncT[nextP]

				total += t

				update(state,t)

				state[nextP] = T2

				isEmpty[nextP] = 0	

			else:														 # 如果没有空闲

				f.write("第{}个物料的工序二下料开始时间为{}\tCNC编号为{}号\n".format(log[nextP],total,nextP+1))

				f.write("第{}个物料的工序二上料开始时间为{}\tCNC编号为{}号\n".format(rgv,total,nextP+1))

				if state[nextP] > 0:									 # 如果还在工作就等待结束

					t = state[nextP]

					total += t

					update(state,t)

				t = cncT[nextP]+Tc

				total += t

				update(state,t)

				state[nextP] = T2

			log[nextP] = rgv

			rgv = 0

		currP = nextP

		temp = total 

		index += 1	

	f.close()

	total += tm[currP][Type.index(0)]									 # 最后归到起始点

	return count1,rgv,currP,total

 

def time_calc(seq,state,isEmpty,rgv,currP,total):						 # 主要用于记录时间

	index = 0

	temp = 0

	while index<len(seq):

		nextP = seq[index]

		t = tm[currP][nextP]

		total += t

		update(state,t)

		if Type[nextP]==0:												 # 如果下一个位置是第一道工作点

			if rgv==1:													 # 然而载着半成品

				seq.pop(index)											 # 去掉这个元素并中止当次循环进入下一个循环

				continue				

			if isEmpty[nextP]:											 # 如果下一个位置是空的

				t = cncT[nextP]

				total += t

				update(state,t)

				state[nextP] = T1										 # 更新当前的CNC状态

				isEmpty[nextP] = 0										 # 就不空闲了

			else:														 # 如果没有空闲

				if state[nextP] > 0:									 # 如果还在工作就等待结束

					t = state[nextP]

					total += t

					update(state,t)

				t = cncT[nextP]											 # 完成一次上下料

				total += t

				update(state,t)

				state[nextP] = T1

				rgv = 1

		else:															 # 如果下一个位置是第二道工作点

			if rgv==0:													 # 如果是个空车

				seq.pop(index)											 # 删除当前节点

				continue

			if isEmpty[nextP]:											 # 如果下一个位置是空的

				t = cncT[nextP]

				total += t

				update(state,t)

				state[nextP] = T2

				isEmpty[nextP] = 0	

			else:														 # 如果没有空闲

				if state[nextP] > 0:									 # 如果还在工作就等待结束

					t = state[nextP]

					total += t

					update(state,t)

				t = cncT[nextP]+Tc

				total += t

				update(state,t)

				state[nextP] = T2

			rgv = 0

		currP = nextP

		temp = total 

		index += 1	

	return rgv,currP,total

 

def forward1(state,isEmpty,currP):										 # 一步最优

	lists = []

	if currP in A:

		rgv = 1

		for e1 in B:

			lists.append([e1])

	

	else:

		rgv = 0

		for e1 in A:

			lists.append([e1])

	

	minV = 28800

	for i in range(len(lists)):

		t = time_calc(lists[i],state[:],isEmpty[:],rgv,currP,0)[-1]

		if t<minV:

			minV = t

			index = i

	return lists[index][0]

 

def forward4(state,isEmpty,currP):										 # 四步最优

	lists = []

	""" 遍历所有的可能性 """

	if currP in A:														 # 如果当前在第二道工序CNC的位置

		rgv = 1

		for e1 in B:

			for e2 in A:

				for e3 in B:

					for e4 in A:

						lists.append([e1,e2,e3,e4])

	else:

		rgv = 0

		for e1 in A:

			for e2 in B:

				for e3 in A:

					for e4 in B:

						lists.append([e1,e2,e3,e4])

	minV = 28800

	for i in range(len(lists)):

		t = time_calc(lists[i],state[:],isEmpty[:],rgv,currP,0)[-1]

		if t<minV:

			minV = t

			index = i

	return lists[index][0]												 # 给定下一步的4步计算最优

 

def forward5(state,isEmpty,currP):										 # 五步最优

	lists = []

	""" 遍历所有的可能性 """

	if currP in A:														 # 如果当前在第二道工序CNC的位置

		rgv = 1

		for e1 in B:

			for e2 in A:

				for e3 in B:

					for e4 in A:

						for e5 in B:

							lists.append([e1,e2,e3,e4,e5])

	else:

		rgv = 0

		for e1 in A:

			for e2 in B:

				for e3 in A:

					for e4 in B:

						for e5 in A:

							lists.append([e1,e2,e3,e4,e5])

	minV = 28800

	for i in range(len(lists)):

		t = time_calc(lists[i],state[:],isEmpty[:],rgv,currP,0)[-1]

		if t<minV:

			minV = t

			index = i

	return lists[index][0]												 # 给定下一步的5步计算最优

 

def forward6(state,isEmpty,currP):										 # 六步最优

	lists = []

	""" 遍历所有的可能性 """

	if currP in A:														 # 如果当前在第二道工序CNC的位置

		rgv = 1

		for e1 in B:

			for e2 in A:

				for e3 in B:

					for e4 in A:

						for e5 in B:

							for e6 in A:

								lists.append([e1,e2,e3,e4,e5,e6])

	else:

		rgv = 0

		for e1 in A:

			for e2 in B:

				for e3 in A:

					for e4 in B:

						for e5 in A:

							for e6 in B:

								lists.append([e1,e2,e3,e4,e5,e6])

	minV = 28800

	for i in range(len(lists)):

		t = time_calc(lists[i],state[:],isEmpty[:],rgv,currP,0)[-1]

		if t<minV:

			minV = t

			index = i

	return lists[index][0]												 # 给定下一步的6步计算最优

 

def forward7(state,isEmpty,currP):										 # 七步最优

	lists = []

	""" 遍历所有的可能性 """

	if currP in A:														 # 如果当前在第二道工序CNC的位置

		rgv = 1

		for e1 in B:

			for e2 in A:

				for e3 in B:

					for e4 in A:

						for e5 in B:

							for e6 in A:

								for e7 in B:

									lists.append([e1,e2,e3,e4,e5,e6,e7])

	else:

		rgv = 0

		for e1 in A:

			for e2 in B:

				for e3 in A:

					for e4 in B:

						for e5 in A:

							for e6 in B:

								for e7 in A:

									lists.append([e1,e2,e3,e4,e5,e6,e7])

	minV = 28800

	for i in range(len(lists)):

		t = time_calc(lists[i],state[:],isEmpty[:],rgv,currP,0)[-1]

		if t<minV:

			minV = t

			index = i

	return lists[index][0]												 # 给定下一步的7步计算最优

 

def forward8(state,isEmpty,currP):										 # 八步最优

	lists = []

	""" 遍历所有的可能性 """

	if currP in A:														 # 如果当前在第二道工序CNC的位置
		rgv = 1

		for e1 in B:
			for e2 in A:
				for e3 in B:
					for e4 in A:
						for e5 in B:
							for e6 in A:
								for e7 in B:
									for e8 in A:
										lists.append([e1,e2,e3,e4,e5,e6,e7,e8])

	else:
		rgv = 0

		for e1 in A:
			for e2 in B:
				for e3 in A:
					for e4 in B:
						for e5 in A:
							for e6 in B:
								for e7 in A:
									for e8 in B:
										lists.append([e1,e2,e3,e4,e5,e6,e7,e8])

	minV = 28800

	for i in range(len(lists)):
		t = time_calc(lists[i],state[:],isEmpty[:],rgv,currP,0)[-1]

		if t<minV:
			minV = t
			index = i

	return lists[index][0]												 # 给定下一步的8步计算最优

def greedy(state,isEmpty,rgv,currP,total):								 # 贪婪算法
	line = []
	count = 0

	while True:
		#nextP = forward4(state[:],isEmpty[:],currP)		
		nextP = forward5(state[:],isEmpty[:],currP)		
		line.append(nextP)
		rgv,currP,t = time_calc([nextP],state,isEmpty,rgv,currP,0)
		total += t
		count += 1

		if total>=28800:
			break

	return line

if __name__ == "__main__":
	state,isEmpty,log,count1,rgv,currP,total,seq = init_first_round()
    
	print(state,isEmpty,log,count1,rgv,currP,total,seq)
    
	line = greedy(state[:],isEmpty[:],rgv,currP,total)
	simulate(line,state,isEmpty,log,count1,rgv,currP,total)
#write_xls()
print("OK")
















