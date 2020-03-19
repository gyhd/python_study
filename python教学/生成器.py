# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 18:32:06 2019

@author: Maibenben
"""

def gen():
    value= 0
    while True:
        receive = yield value
        if receive == 'e':
            break
        value = 'got: %s'%receive

g = gen()
print(g.send(None))
#print(g.send('aaa'))
#print(g.send(3))
#print(g.send('e'))


def gen():
    while True:
        try:
            yield 'normal value'
            yield 'normal value 2'
            print('here')
        except ValueError:
            print('we got ValueError here')
        except TypeError:
            break
g = gen()
print(next(g))
print(g.throw(ValueError))
print(next(g))
print(next(g))
#print(g.throw(TypeError))

import logging

def use_logging(func):
    def wrapper(*args,**kwargs):
        
        logging.warning("%s is running" %func.__name__)
        return func(*args,**kwargs)
    return wrapper
def use_logging1(func):
    def hhh(*args,**kwargs):
        
        logging.warning("%s is shopping" %func.__name__)
        return func(*args,**kwargs)
    return hhh

@use_logging    
@use_logging1
def foo():
    print('Hi,Python')
foo()
#foo()=use_logging(use_logging1(foo))

import logging
def deco(tag):
    def use_logging(func):
        def warpper(*args,**kwargs):
            logging.warning("%s is running , this is %s" %(func.__name__,tag))
            return func(*args,**kwargs)
        return warpper
    return use_logging

@deco('python')
def bar():
    print('i am foo')
bar()





import logging

def use_logging(func):
    def wrapper(*args,**kwargs):
        for i in range([foo]):
            if i%1 != 0:
                logging.warning("%s is running" %func.__name__)
                break
        return func(*args,**kwargs)
    return wrapper
@use_logging
def foo(a,b,c,d,e,f,g):
    print('this is a camulate')
    print(a+b+c+d+e+f+g)
foo(5,6,9,8,4,5,1)








































