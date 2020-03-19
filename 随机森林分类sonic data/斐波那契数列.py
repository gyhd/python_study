# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 22:45:13 2019

@author: Maibenben
"""

f1 = 1
f2 = 1
print(f1,'',f2,'',end='')
for i in range(3,101):
    f3 = f1 + f2
    if f3 > 100:
        break
    f1 = f2
    f2 = f3
    print(f3,'',end='')
