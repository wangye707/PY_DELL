#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : numpy_reshape.py
# @Author: WangYe
# @Date  : 2018/7/23
# @Software: PyCharm
import numpy
a=[1,2,3,1,4,5,5846,464,64,456,456,4]
print(a)
b=numpy.array(a).reshape(len(a),1)   # reshape(列的长度，行的长度)
print(b)  #转换为二维矩阵
print('b的形状是'+numpy.shape(b))  #12行1列