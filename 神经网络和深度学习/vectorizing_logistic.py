# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 19:21:29 2019

@author: Maibenben
"""

import numpy as np

A = np.array([[1,2,3],
              [1,5,6],
              [5,6,8],
              [7,1,3]])

print ('A=',A)

cal = A.sum(axis=0)#纵轴方向为0，横轴方向为1
print(cal)


percentage = A/cal.reshape(1,3)#reshape可要可不要，主要作用是确定被除数
print(percentage)

"""
B = np.random.randn(5)

print(B)
print(B.shape)
print(B.T)
print(np.dot(B,B.T))
"""

B = np.random.randn(5,1)#编程时使用此类，不要用上面那种

print(B)
print(B.shape)
print(B.T)
print(np.dot(B,B.T))#求两者的内积


costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()






