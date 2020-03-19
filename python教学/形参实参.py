# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:29:16 2019

@author: Maibenben
"""

b=[1]
def changer(a):
    global b
    a.append(2)

changer(b)
print(b)


for i in range(1,10):
    for j in range(1,i+1):
        print("%d * %d = %d"%(i,j,i*j),end=',')
print()

"""
def word(name,greeting = 'hi'):
    print(greeting,name +'!')
word('hello','jack')


def word(name,*,greeting,name1):
    print(greeting,name + '!',name1)
word(name1='jack',Name='gg',name='hello',greeting='fiona')


def f1(a,b,c=0,*args,**kw):
    print('a=',a,'b=',b,'c=',c,'args=',args,'kw=',kw)

def f2(a,b,c=0,*,d,**kw):
    print('a=',a,'b=',b,'c=',c,'d=',d,'kw=',kw)
    
f1(1,2,3,'a','b',x=99)
f2(1,2,d=99,ext=None)
"""







