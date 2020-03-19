# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 18:46:34 2019

@author: Maibenben
"""


import os

os.environ["PATH"] += os.pathsep + 'E:/anaconda/release/bin/'  #注意修改你的路径




from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pydotplus
import graphviz


iris = load_iris()
clf = DecisionTreeClassifier()
clf_iris = clf.fit(iris.data,iris.target)
dot_Data = export_graphviz(clf_iris,out_file=None)

graph_iris = pydotplus.graph_from_dot_data(dot_Data)
graph_iris.write_pdf("./load_iris.pdf")




