# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 19:36:47 2019

@author: Maibenben
"""

from keras.datasets import mnist
from keras.utils import to_categorical

train_X, train_y = mnist.load_data()[0]
train_X = train_X.reshape(-1, 28, 28, 1)
train_X = train_X.astype('float32')
train_X /= 255
train_y = to_categorical(train_y, 10)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta

model = Sequential()
model.add(Conv2D(32, (5,5), activation='relu', input_shape=[28, 28, 1]))
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=categorical_crossentropy,
             optimizer=Adadelta(),
             metrics=['accuracy'])

#epoch中文叫做新纪元，每经过1次epoch，即模型训练遍历所有样本1次；
#上文中epoch设置为1，即模型训练遍历所有样本1次；
#batch_size设置为100，即每次模型训练使用的样本数量为100；
#每经过1次epoch，模型遍历训练集的n个样本，每次训练使用100个样本，
#即模型训练n/100次，即损失函数经过次批量梯度下降。
batch_size = 100
epochs = 1
model.fit(train_X, train_y,
         batch_size=batch_size,
         epochs=epochs)

test_X, test_y = mnist.load_data()[1]
test_X = test_X.reshape(-1, 28, 28, 1)
test_X = test_X.astype('float32')
test_X /= 255
test_y = to_categorical(test_y, 10)

loss, accuracy = model.evaluate(test_X, test_y, verbose=1)

print('loss:%.4f accuracy:%.4f' %(loss, accuracy))





