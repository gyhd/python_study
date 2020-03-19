# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:03:59 2019

@author: Maibenben
"""

list1=[3,5,4,-2,0,-1]
sorted(list1, key=lambda x: abs(x))

multiply=(lambda x, y: x*y)#括号可要可不要
print(multiply(5,6))


def get_y(a,b):
    return (lambda x: a*x+b)#括号可要可不要

a= get_y(2,5)
print(a(2))

y= map(lambda x:x*x, [y for y in range(10)])
print(y)

list(map( lambda x: x*x, [y for y in range(10)]))
#加了list后才能改成我们想要的格式,而且必须要有map


def formate_name(s):
    s1 = s[0].upper()
    s2 = s[1:].lower()
    return s1 + s2
 
print (list(map(formate_name , ['adam', 'LISA', 'barT'])))


