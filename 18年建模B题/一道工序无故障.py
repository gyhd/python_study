# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:12:22 2019
@author: Maibenben
"""
"""
	作者：囚生CY
	平台：CSDN
	时间：2018/10/09
	转载请注明原作者
	创作不易，仅供分享
"""

 

import math
import random
import itertools


"""

#第二组
d1 = 23
d2 = 41
d3 = 59
Te = 35
To = 30
Tc = 30

T = 580

"""

"""
# 第3组
d1 = 18
d2 = 32
d3 = 46
To = 27
Te = 32
Tc = 25

T = 545
"""



# 第1组


d1 = 20
d2 = 33
d3 = 46
To = 28
Te = 31
Tc = 25

T = 560





CNCT = [To,Te,To,Te,To,Te,To,Te]										 # CNC上下料时间

N = 50
L = 17
varP = 0.1
croP = 0.6
croL = 4
e = 0.99


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

def update_state(state,t):
	length = len(state)
	for i in range(length):
		if state[i] < t:
			state[i] = 0
		else:
			state[i] -= t
	return state


def time_calc(seq):
	state = [0 for i in range(8)] 			# 记录CNC状态
	isEmpty = [1 for i in range(8)]			 # CNC是否为空？
	currP = 0
	total = 0
	length = len(seq)

	for No in seq:
		nextP = No
		t = tm[currP][nextP]
		total += t														 # rgv移动
		state = update_state(state,t)									 # 更新state
		if state[No]==0:												 # 表明CNC等待
			if isEmpty[No]:												 # 当前CNC空
				t = CNCT[No]
				isEmpty[No] = 0
			else:
				t = CNCT[No]+Tc
			total += t
			state = update_state(state,t)
			state[No] = T
		else:															 # 当前CNC忙
			total += state[No]											 # 先等当前CNC结束
			state = update_state(state,state[No])						 
			t = CNCT[No]+Tc
			total += t
			state = update_state(state,t)
			state[No] = T
		currP = No
	total += tm[currP][0]
	return total

def init_prob(sample):
	prob = []
	for seq in sample:
		prob.append(time_calc(seq))
	maxi = max(prob)
	prob = [maxi-prob[i]+1 for i in range(N)]
	temp = 0
    
	for p in prob:
		temp += p
	prob = [prob[i]/temp for i in range(N)]
    
	for i in range(1,len(prob)):
		prob[i] += prob[i-1]
	prob[-1] = 1			 # 精度有时候很出问题
	return prob

 

def minT_calc(sample):
	minT = time_calc(sample[0])
	index = 0
	for i in range(1,len(sample)):
		t = time_calc(sample[i])
		if t < minT:
			index = i
			minT = t
	return minT,index

def init():
	sample = []
	for i in range(N):
		sample.append([])

		for j in range(L):
			sample[-1].append(random.randint(0,7))

	return sample


def select(sample,prob):												 # 选择
	sampleEX = []
	for i in range(N):													 # 取出N个样本
		rand = random.random()
		for j in range(len(prob)):
			if rand<=prob[j]:
				sampleEX.append(sample[j])
				break
	return sampleEX

 

def cross(sample,i):													 # 交叉
	for i in range(len(sample)-1):
		for j in range(i,len(sample)):
			rand = random.random()
			if rand<=croP*(e**i):										 # 执行交叉
				loc = random.randint(0,L-croL-1)
				temp1 = sample[i][loc:loc+croL]
				temp2 = sample[j][loc:loc+croL]
				for k in range(loc,loc+croL):
					sample[i][k] = temp2[k-loc]
					sample[j][k] = temp1[k-loc]
	return sample

		

def variance(sample,i):													 # 变异算子										 
	for i in range(len(sample)):
		rand = random.random()
		if rand<varP*(e**i):
			rand1 = random.randint(0,L-1)
			rand2 = random.randint(0,L-1)
			temp = sample[i][rand1]
			sample[i][rand1] = sample[i][rand2]
			sample[i][rand2] = temp
	return sample
	

def main():
	sample = init()
	mini,index = minT_calc(sample)
	best = sample[index][:]
	print(best)

	for i in range(10000):
		print(i,'\t',minT_calc(sample),end="\t")
		prob = init_prob(sample)
		sample = select(sample,prob)
		sample = cross(sample,i)
		sample = variance(sample,i)
		mi,index = minT_calc(sample)
		if mi>mini and random.random()<e**i:			 # 精英保留策略
			rand = random.randint(0,N-1)
			sample[rand] = best[:]
		mini,index = minT_calc(sample)
		best = sample[index][:]
		print(best)
	print(sample)

if __name__ == "__main__":
	""" 穷举搜索验证 """
	a = list(itertools.permutations([1,2,3,4,5,6,7],7))
	ts = []
	first = [0,1,2,3,4,5,6,7,0]

	for i in a:
		temp = first+list(i)
		temp.append(0)
		t = time_calc(temp)
		ts.append(t)

	print(min(ts))	
	print(time_calc([0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0]))







