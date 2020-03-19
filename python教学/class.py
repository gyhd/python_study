# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:47:23 2019

@author: Maibenben
"""


class Grandfather:
    def __init__(self):
        print('I\'m grandfather')
        
class Father(Grandfather):
    def __init__(self):
        print('I\'m father')
        
class Son(Father):
    def __init__(self):
        print('This is constracted function,Son')
    def sayHello(self):
        return 'Hello python'

if __name__=='__main__':
    son=Son()
    #打印出来类型帮助信息
    print('类型帮助信息:',Son.__doc__)
    #打印出来类型名称
    print('类型名称:',Son.__name__)
    #打印出来类型字典
    print('类型字典:',Son.__dict__)
    #打印出来所继承的基类
    print('继承的基类:',Son.__bases__)
    #打印出来类型模块
    print('类型模块:',Son.__module__)
    #打印出来实例类型
    print('实例类型:',Son.__class__)

class Calculator:
    name='calculator'#固有属性
    price=28
    
    #添加参数，里面可以进一步修改
    def __init__(self,name,hight,width,weight,price):
        self.name=name
        self.price=price
        self.h=hight
        self.w=width
        self.weight=weight
        
    def plus(self,x,y):
        print(self.name)
        print(x+y)
    def subtract(self,x,y):
        print(x-y)
    def multiply(self,x,y):
        print(x*y)
    def divide(self,x,y):
        print(x/y)
        print(self.price)

c=Calculator('cat',100,200,300,26)
print(c.divide(6,2))
print(c.plus(6,2))
print(c.multiply(6,2))
print(c.subtract(6,2))

print(c.name)
print(c.weight)
print(c.price)















