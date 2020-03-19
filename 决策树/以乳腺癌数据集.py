# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 20:14:46 2019

@author: Maibenben
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

import numpy as np
import matplotlib.pyplot as plt
import pydotplus
import graphviz


iris = load_iris()

cancer = load_breast_cancer()
clf = DecisionTreeClassifier(max_depth=7)
clf_iris = clf.fit(iris.data,iris.target)
clf_cancer = clf.fit(cancer.data,cancer.target)

print("cancer.keys(): \n{}".format(cancer.keys()))
print(cancer.target)
print("{}".format({n:v for n,v in zip(cancer.target_names,np.bincount(cancer.target))}))

x_train,x_test,y_train,y_test = train_test_split(cancer.data,cancer.target,test_size=0.3,random_state=1)
clf_DT  = clf.fit(x_train,y_train)

print("{:.3f}".format(clf_DT.score(x_train,y_train)))
print("{:.3f}".format(clf_DT.score(x_test,y_test)))

export_graphviz(clf_DT,out_file="./DT.dot",class_names=['malignant','benign'],feature_names=cancer.feature_names,impurity=False,filled=True)
print("{}".format(clf_DT.feature_importances_))
print(cancer.data.shape[0])

def polt_feature_importances_cancer(model):
    feature_count = cancer.data.shape[1]
    plt.barh(range(feature_count),model.feature_importances_,align='center')
    plt.yticks(np.arange(feature_count),cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()

polt_feature_importances_cancer(clf_DT)




