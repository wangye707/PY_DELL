#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : packageTest.py
# @Author: WangYe
# @Date  : 2018/3/8
# @Software: PyCharm
#测试Scikit-learn
# from sklearn import datasets
# iris=datasets.load_iris()
# digits=datasets.load_iris()
# print(type(digits.data))
#测试是否包含网络分析库
# import networkx as  nx
# G=nx.Graph()
# G.add_node(1)
# print(type(G))
#自然语言工具包
#import nltk
#nltk.download()
from nltk import brown
print(brown.words)()[0:10]