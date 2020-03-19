# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 23:02:22 2019

@author: Maibenben
"""

score = input('please into a number:')
totalscore = 0
while True:
    if (score == "done"):
        break
    b = int(score)
    if (b > 100) | (b < 0):
        print('the number you input is wrong')
        score = input('please into a number:')
        continue
    else:
        totalscore += b
        score = input('please into a number:')
print('the totalscore is',totalscore)

