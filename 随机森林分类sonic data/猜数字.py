# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 22:36:53 2019

@author: Maibenben


import random
a = random.randint(1,100)
b = int(input('please into a number:'))

while a != b:
    if a > b:
        print('you guess smaller')
        b = int(input('please into a number:'))
    elif a < b:
        print('you guess bigger')
        b = int(input('please into a number:'))

print('you guess right')

"""

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import linear_model

clf = linear_model.LogisticRegression()
# X:features  y:targets  cv:k
cross_val_score(clf, X, y, cv=5)

