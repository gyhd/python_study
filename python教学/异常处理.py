# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:16:13 2019

@author: Maibenben
"""

try:
    a=int(input('a number:'))
#如果是下列两种错误类型，则打印出value is wrong
except(ValueError,SyntaxError):
    print('value is wrong')
except:
    print('wrong')

else:
    print('the number is %f'%a)
finally:
    print('input is done')


try:
    print('aa')#可以执行
    raise NameError('Hi,python')
    print('bb')#不可以执行
except NameError:
    print('An exception flew by')


class MyError(Exception):
    print('vv')
    pass
try:
    raise MyError()
except MyError:
    print('an error')#先打印出来vv，再打印an error


def divide():
    flag = True
    while(flag):
        try:
            x=list(map(int("input two numbers,with blank split:").split()))
            result = x[0] / x[1]
            flag = False
        except(ZeroDivisionError,ValueError):
            print('input wrong,input again:')
        else:
            print('result is:',result)
        finally:
            print('calculate is done')
            
divide()



def divide(x,y):
    assert y != 1,'y is zero'
    return (x/y)

print (divide(5,1))
























