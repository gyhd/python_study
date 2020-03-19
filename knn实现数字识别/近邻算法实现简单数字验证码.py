# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:53:26 2019

@author: Maibenben
"""

from numpy import *
import operator
from os import listdir

def knn(k,testdata,traindata,labels):
    #testdata:[特征1，特征2，特征3]
    #traindata:[[特征1，特征2，特征3],[特征1，特征2，特征3],[特征1，特征2，特征3]]
    traindatasize=traindata.shape[0]
    dif=tile(testdata,(traindatasize,1))-traindata
    sqdif=dif**2

    #sumsqdif已经成为一维的了[a,b,c,d]
    sumsqdif=sqdif.sum(axis=1)
    distance=sumsqdif**0.5
    sortdistance=distance.argsort()

    #sortdistance指的是测试数据与各训练数据的距离由近到远排序之后的结果列表
    count={}#{"类别":"次数"}

    for i in range(0,k):
        vote=labels[sortdistance[i]]#当前距离的类别是谁（由近至远）
        count[vote]=count.get(vote,0)+1

    #print(count)
    sortcount=sorted(count.items(),key=operator.itemgetter(1),reverse=True)
    return sortcount[0][0]

#数据加载，将数据转换成knn算法支持的向量。
def datatoarray(fname):
    arr=[]
    fh=open(fname)
    for i in range(0,32):
        thisline=fh.readline()
        for j in range(0,32):
            arr.append(int(thisline[j]))
    return arr

#a=datatoarray("D:/Python35/traindata/0_3.txt")
#取文件名前缀（类别）
def seplabel(fname):
    filestr=fname.split(".")[0]
    label=int(filestr.split("_")[0])
    return label

#建立训练数据
#labels:[类别,类别，类别,类别]
#tainarr：[[特征1，特征2，特征3],[特征1，特征2，特征3],[特征1，特征2，特征3]]
def traindata():
    labels=[]  #存放类别
    trainfile=listdir("./trainandtest/traindata") #把文件夹打开并且把文件夹中所
    #有的文件名存放在列表中，为了最后放在datatoarray函数的arr里（那里面也是列表类型）
    num=len(trainfile)
    
    #列为1024,行为num的数组
    trainarr=zeros((num,1024))

    for i in range(0,num):
        thisname=trainfile[i]
        thislabel=seplabel(thisname)
        labels.append(thislabel)
        trainarr[i,:]=datatoarray("./trainandtest/traindata/"+thisname)
    return trainarr,labels   

trainarr,labels=traindata()
thistestfile="4_91.txt"
testarr=datatoarray("./trainandtest/testdata/"+thistestfile)
rst=knn(3,testarr,trainarr,labels)

print ("this writing's classify is",rst)

#小作业
#实现将testdata里面的数据批量识别（提示：通过循环）
#2：实现计算识别准确率SA




