# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:43:36 2019

@author: Maibenben


import functools

def int8(x):
    return functools.partial(int,base=8)

print(int8('12'))


with open('a.txt','a') as f:
    f.write('hahahah')





a = ['a','b','c',2,3,4]
group_ad=lambda x,k : zip(*([iter(x)]*k))

for i in group_ad(a,3):
    print(i)
"""



"""
def yunnian(x):
    return (x%400 == 0 or (x%100 == 0 and x%4 != 0))
m = [0,31,28,31,30,31,30,31,31,30,31,30,31]
yes = 0
year = int(input('year:'))
month = int(input('month:'))
day = int(input('day:'))

if yunnian(year):
    m[2] += 1
for i in range(month):
    yes += m[i]
print(yes+day)
"""
"""

def isLeapYear(y):
    return (y%400==0 or (y%4==0 and y%100!=0))
DofM=[0,31,28,31,30,31,30,31,31,30,31,30]
res=0
year=int(input('Year:'))
month=int(input('Month:'))
day=int(input('day:'))
if isLeapYear(year):
    DofM[2]+=1
for i in range(month):
    res+=DofM[i]
print(res+day)



"""
"""

w = [1,2,3,'6','b','c']
group = lambda x,k : zip(*([iter(x)]*k))

for i in group(w,2):
    print(i)



a = ['a','b','c',2,3,4]
group_ad=lambda x,k : zip(*([iter(x)]*k))

for i in group_ad(a,3):
    print(i)




def fib(max):
    n,a,b = 0,0,1
    while n < max:
        yield b
        a,b = b,a+b
        n += 1
    return 'done'

for i in fib(5):
    print(i)
    """
    
    
import os
import shutil

def findfile():
    filename ='d:/tmp'
    if os.path.exists(filename):
        print('ok'+filename+'exists')
    else:
        shutil.copytree('d:/亚太赛建模/回归分析','d:/tmp')
        print('completed')

findfile()














