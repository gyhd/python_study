# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:20:16 2019

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


"""
#第3组
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
Type = [1,0,1,0,1,0,0,0]												 # CNC刀具分类

N = 64
L = 100
varP = 0.1
croP = 0.6
croL = 2
e = 0.99

def init_first_round():													 # 第一圈初始化（默认把所有第一道CNC按顺序加满再回到当前位置全部加满）
	state = [0 for i in range(8)] 									 	 # 记录CNC状态（还剩多少秒结束，0表示空闲）
	isEmpty = [1 for i in range(8)]										 # CNC是否为空
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
	rgv,currP,total = time_calc(seq,state,isEmpty,rgv,currP,total)

	return state,isEmpty,rgv,currP,total,seq

def update(state,t):

	for i in range(len(state)):
		if state[i] < t:
			state[i] = 0
		else:
			state[i] -= t

def time_calc(seq,state,isEmpty,rgv,currP,total):						 # 事实上sequence可能是无效的，所以可能需要
	index = 0
	temp = 0

	while index<len(seq):
		""" 先移动到下一个位置 """
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
	total += tm[currP][Type.index(0)]									 # 最后归零

	return rgv,currP,total

def init_prob(sample,state,isEmpty,rgv,currP,total):					 # 计算所有sample的
	prob = []

	for seq in sample:
		t = time_calc(seq,state[:],isEmpty[:],rgv,currP,total)[-1]
		prob.append(t)
	maxi = max(prob)
	prob = [maxi-prob[i]+1 for i in range(N)]
	temp = 0

	for p in prob:
		temp += p
	prob = [prob[i]/temp for i in range(N)]

	for i in range(1,len(prob)):
		prob[i] += prob[i-1]
	prob[-1] = 1														 # 精度有时候很出问题

	return prob

def minT_calc(sample,state,isEmpty,rgv,currP,total):
	minT = time_calc(sample[0],state[:],isEmpty[:],rgv,currP,total)[-1]
	index = 0

	for i in range(1,len(sample)):
		t = time_calc(sample[i],state[:],isEmpty[:],rgv,currP,total)[-1]
		if t < minT:
			index = i
			minT = t

	return minT,index

def init():						 # 初始化种群（按照第二道工序，第一道工序，第二道工序，第一道工序顺序排列即可）
	sample = []
	refer0 = []
	refer1 = []
    
	for i in range(8):

		if Type[i]==0:
			refer0.append(i)
		else:
			refer1.append(i)

	for i in range(N):
		sample.append([])
		for j in range(L):
			if j%2==0:
				sample[-1].append(refer1[random.randint(0,len(refer1)-1)])
			else:
				sample[-1].append(refer0[random.randint(0,len(refer0)-1)])
	return sample

 

def select(sample,prob):												 # 选择算子
	sampleEX = []
	for i in range(N):													 # 取出N个样本
		rand = random.random()
		for j in range(len(prob)):
			if rand<=prob[j]:
				sampleEX.append(sample[j])
				break
	return sampleEX

 

def cross(sample,i):						 # 交叉算子
	for i in range(len(sample)-1):
		for j in range(i,len(sample)):
			rand = random.random()

			if rand<=croP*(e**i):				 # 执行交叉
				loc = random.randint(0,L-croL-1)
				temp1 = sample[i][loc:loc+croL]
				temp2 = sample[j][loc:loc+croL]

				for k in range(loc,loc+croL):
					sample[i][k] = temp2[k-loc]
					sample[j][k] = temp1[k-loc]
	return sample

		

def variance(sample,i):				 # 变异算子										 
	for i in range(len(sample)):
		rand = random.random()
		if rand<varP*(e**i):
			rand1 = random.randint(0,L-1)
			randTemp = random.randint(0,int(L/2)-1)
			rand2 = 2*randTemp if rand1%2==0 else 2*randTemp+1
			temp = sample[i][rand1]
			sample[i][rand1] = sample[i][rand2]
			sample[i][rand2] = temp
	return sample

if __name__ == "__main__":
	state,isEmpty,rgv,currP,total,seq = init_first_round()
	print(state,isEmpty,rgv,currP,total)

	sample = init()
	mini,index = minT_calc(sample,state[:],isEmpty[:],rgv,currP,total)	
	best = sample[index][:]

	for i in range(100000):
		f = open("GA.txt","a")
		tmin = minT_calc(sample,state[:],isEmpty[:],rgv,currP,total)[0]
		f.write("{}\t{}\n".format(i,tmin))
		print(i,"\t",tmin,end="\t")
		prob = init_prob(sample,state[:],isEmpty[:],rgv,currP,total)
		sample = select(sample,prob)
		sample = cross(sample,i)
		sample = variance(sample,i)
		mi,index = minT_calc(sample,state[:],isEmpty[:],rgv,currP,total)

		if mi>mini and random.random()<e**i:				 # 精英保留策略
			rand = random.randint(0,N-1)
			sample[rand] = best[:]
		mini,index = minT_calc(sample,state[:],isEmpty[:],rgv,currP,total)
		best = sample[index][:]
		print(best)
		f.close()

	print(sample)





