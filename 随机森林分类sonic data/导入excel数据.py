# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:08:26 2019

@author: Maibenben
"""
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
import matplotlib.pylab as plt
import matplotlib.pyplot as plt

#导入数据，顺便看看数据的类别分布
train= pd.read_excel('C:\\Users\\Maibenben\\Desktop\\2019MathorCup\\2019MathorCup\\D\\E1.xlsx')

target='Disbursed' # Disbursed的值就是二元分类的输出
IDcol= 'ID'

train['转炉终点C'].value_counts()
b = train['转炉终点C'].value_counts()
print(b)

#接着选择好样本特征和类别输出，样本特征为除去ID和输出类别的列
x_columns = [x for x in train.columns if x not in [target,IDcol]]
X = train[x_columns]
y = train['转炉终点C'] 
#不管任何参数，都用默认的，拟合下数据看看
rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(X,y)
print (rf0.oob_score_)
y_predprob = rf0.predict_proba(X)[:,1]
print ("AUC Score (Train): %s") % metrics.roc_auc_score(y, y_predprob)


param_test1= {'n_estimators':range(10,71,10)}
gsearch1= GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
        min_samples_leaf=20,max_depth=8,max_features='sqrt',random_state=10),
                       param_grid =param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(X,y)
gsearch1.grid_scores_,gsearch1.best_params_, gsearch1.best_score_

