# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:48:22 2019

@author: Maibenben
"""
"""

def use_logging(func):
    def wrapper(*args,**kwargs):
        kwarg_values=[i for i in kwarg_values()]
        for arg in list(args) + kwarg_values:
            if not isinstance(arg,int):
                return print('wrong input')
        return func(*args,**kwargs)
    return wrapper

@use_logging
def foo(a,b):
    return (a+b)
foo(5,1)
"""

import hello
print('I am python')


class Foo(object):
    def __init__(self,func):
        self.func=func
        
    def __call__(self):
        print("%s is running"%self.func)
        self.func()
        print("%s is end"%self.func)
        
@Foo
def bar():
    print('bar')
bar()


class people:
    def __init__(self,n,a):
        self.__name=n
        self.__age=a
    @property
    def age(self):
        return print(self.__age)
    @age.setter
    def age(self,age):
        self.__age=age
        
    def speak(self):
        print("%s says: I am %d years old"%(self.__name,self.__age))

#调用实例    
p=people('fiona',20)
p.age=50
p.speak()


class A(object):
    bar=1
    def func1(self):
        print('foo')
    @classmethod
    def func2(cls):
        print('func2')
        print(cls.bar)
        cls().func1() #调用 foo 方法
        
A.func2()  #不需要实例化



class C(object):
    @staticmethod
    def f():
        print('fiona')

C.f()      #静态方法无需实例化
a=C()
a.f()      #也可以实例化后调用


