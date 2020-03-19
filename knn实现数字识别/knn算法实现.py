# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 19:53:13 2019

@author: Maibenben
"""


import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def predict(test_features,train_features,train_labels):

    predict=[]
    count=0
    train_size=len(train_labels)
    test_size=len(test_features)

    for test_vec in test_features:
        print ('predicting:',count,'/',test_size)
        count+=1
        max_index=-1
        dist_max=0
        knn_list = []

        for i in range(k):
            train_vec=train_features[i]
            dist=np.sqrt(np.sum(np.square(test_vec-train_vec)))
            label=train_labels[i]
            knn_list.append((dist,label))

        for i in range(k,train_size):
            label = train_labels[i]
            train_vec = train_features[i]
            dist = np.sqrt(np.sum(np.square(test_vec - train_vec)))

            #find the max value of present knn_list
            if max_index==-1:             # 如果 max_index=-1,则发生了交换，需要重新寻找交换后 knn_list的最大值
                for j in range(k):
                    if knn_list[j][0] > dist_max:
                        dist_max=knn_list[j][0]
                        max_index=j

            # if dist < dist_max,swap them
            if dist < dist_max:
                knn_list[max_index]=(dist,label)
                dist_max=0        # dist_max=0 ,为下次寻找最大值
                max_index=-1

        #class_count=[0 for i in range(k)]
        class_count=np.zeros(k)

        for dist,label in knn_list:                  #notice
            class_count[label]+=1

        max_count=max(class_count)

        for i in range(k):
            if max_count==class_count[i]:
                predict.append(i)              #notice append ()
                break

    return np.array(predict)

k=10

if __name__=='__main__':

    print ('Start reading data...')

    time_1 = time.time()
    raw_data=pd.read_csv('./data/train.csv')
    data=raw_data.values

    imgs=data[::,1::]
    labels=data[::,0]

    train_features,test_features,train_labels,test_labels=train_test_split(imgs,labels,test_size=0.25,random_state=1)

    time_2 = time.time()
    print ('reading  cost ', time_2 - time_1, ' seconds', '\n')

    print ('Start predicting...')

    test_predict = predict(test_features, train_features, train_labels)
    time_4 = time.time()

    print ('predicting cost ', time_4 - time_2, ' seconds', '\n')

    score = accuracy_score(test_labels, test_predict)
    print ("The accruacy socre is ", score)











