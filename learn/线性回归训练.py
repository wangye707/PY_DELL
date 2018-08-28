#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : 线性回归训练.py
# @Author: WangYe
# @Date  : 2018/4/7
# @Software: PyCharm
import sys
import os
import numpy as np
filename=os.chdir('C:\\Users\\wy\\Desktop')
X=[]
Y=[]
f=open('test1.txt','r')
for testnum in f.readlines():
    #testnum = testnum.split("/n")
    testnum=testnum.replace(',',' ')   #分词
    testnum=testnum.split()
    #print(testnum)
    #xt,yt=[float(i) for i in line.split(',')]
    xt=int(testnum[0])    #将文件中的字符型数字转换为int
    #print(xt)
    yt=int(testnum[1])    #文件不能有多余的空格，除非做判别处理
    #print(yt)
    X.append(xt)
    Y.append(yt)
    # print(X)
    # print(Y)
print(X)
print(Y)
X_train=np.array(X[:5]).reshape((5,1))   #前5个作为训练集，后三个作为训练集
Y_train=np.array(Y[:5])#reshape是将数组分成5行一列的二维数组
X_test=np.array(X[5:]).reshape(3,1)
Y_test=np.array(Y[5:])
from sklearn import linear_model  #创建回归对象
#线性回归使用的是最小二乘法
linear_regressor=linear_model.LinearRegression()
linear_regressor.fit(X_train,Y_train)   #利用训练集完成回归对象
import matplotlib.pyplot as plt   #创建可视窗口
y_train_Predict=linear_regressor.predict(X_train)  #查看训练集
plt.figure()
plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,y_train_Predict,color='black')
plt.title('WY')
plt.show()
y_test_Predict=linear_regressor.predict(X_test)#查看预测集
plt.figure()
plt.scatter(X_test,Y_test,color='green')
plt.plot(X_test,y_test_Predict,color='black')
plt.title('WYpredict')
plt.show()
#最小二乘法对异常的处理过于敏感
#采用正则化项的系数作为阀值来消除异常值影响

