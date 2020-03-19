# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 21:44:29 2019

@author: Maibenben
"""
"""
num = 100
def func():
    num = 200
    
func()
print(num)

"""

"""
def func():
    num =100
    num += 100
    print(num)

func()
"""
"""
num = 100
def func():
    global num
    num += 100
    
func()
print('num=',num)

"""
"""
with open('a.txt','b') as f:
    f.write('hahah')


def finalresult(n):
    if n < 0:
        print('input wrong')
    if n == 0:
        return 1
    if n == 1:
        return 1
    else:
        return finalresult(n-2) + finalresult(n-1)
print(finalresult(4))



c=lambda x,y,z:x**3+y**2+z
print(c(2,3,4))



def func():
    x,y = 1,2
    a,b = 1,3
    sua = lambda x,y:x + y
    print(sua)
    sub = lambda a,b:a - b
    print(sub)
    return sua(x,y) + sub(a,b)
func()

f = list(map(lambda x:(x+1) if (x < 5) else x))
i = [y for y in range(10)]
print(i)
list(map(f,i))


def f(x):
    return x*x
def h(func,x):
    return func(x)*func(x)

print(h(f,3))
"""

def f(x):
    return x*x

print(list(map(f,(1,2,3))))

























