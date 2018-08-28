#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : ANNtest.py
# @Author: WangYe
# @Date  : 2018/7/20
# @Software: PyCharm
from TEST.ANN import *

def vectorized_result(j,nclass):
    """离散数据进行one-hot"""
    e = np.zeros((nclass, 1))
    e[j] = 1.0
    return e

def get_format_data(X,y,isTest):
    ndim = X.shape[1]
    nclass = len(np.unique(y))
    inputs = [np.reshape(x, (ndim, 1)) for x in X]
    if not isTest:
        results = [vectorized_result(y,nclass) for y in y]
    else:
        results = y
    data = zip(inputs, results)
    return data

#随机生成数据
from sklearn.datasets import *
np.random.seed(0)
X, y = make_moons(200, noise=0.20)
ndim = X.shape[1]
nclass = len(np.unique(y))

#划分训练、测试集
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.2,random_state=0)

training_data = get_format_data(train_x,train_y,False)
test_data = get_format_data(test_x,test_y,True)

net = Network(sizes=[ndim,10,nclass])
net.SGD(training_data=training_data,epochs=5,mini_batch_size=10,eta=0.1,test_data=test_data)