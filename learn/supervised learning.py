#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : supervised learning.py
# @Author: WangYe
# @Date  : 2018/3/26
# @Software: PyCharm
# 监督学习
import numpy as np
from sklearn import preprocessing
x=np.random.rand(4,4)#求4*4的随机列表
z=np.mat(x).I#转换成矩阵，I为求逆
from sklearn import preprocessing
data=np.array([[3,-1.5,2,-5.4],[0,4,-0.3,2.1],[1,3.3,-1.9,-4.3]])
data_standardized=preprocessing.scale(data)
print(data)
# import numpy as np
# x=np.random.rand(4,4)


