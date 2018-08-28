#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : pickle_test.py
# @Author: WangYe
# @Date  : 2018/5/19
# @Software: PyCharm
import pickle
import jieba
#from numpy import *
dataList = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
dataList2 = [[1, 1, 's'],
            [1, 1, 's'],
            [1, 0, 's'],
            [0, 1, 'no'],
            [0, 1, 'no']]
dataDic = {0: [1, 2, 3, 4],
           1: ('a', 'b'),
           2: {'c': 'yes', 'd': 'no'}}

# 使用dump()将数据序列化到文件中
fw = open('dataFile.txt', 'wb')
# Pickle the list using the highest protocol available.
pickle.dump(dataList, fw, -1)
pickle.dump(dataList2, fw, -1)
# Pickle dictionary using protocol 0.
pickle.dump(dataDic, fw)
fw.close()

#使用load()将数据从文件中序列化读出
fr = open('dataFile.txt', 'rb')
data1 = pickle.load(fr)
print(data1)
data2 = pickle.load(fr)       #没有s
print(data2)                  #每次读取一个序列
data3 = pickle.load(fr)
print(data3)
fr.close()

#使用dumps()和loads()举例
p = pickle.dumps(dataList)    #加S是有参数（对象）的载入或者读出
print(pickle.loads(p))
p = pickle.dumps(dataDic)
print(pickle.loads(p))