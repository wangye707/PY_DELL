# #!D:/workplace/python
# # -*- coding: utf-8 -*-
# # @File  : test.py
# # @Author: WangYe
# # @Date  : 2018/7/24
# # @Software: PyCharm
# import pymysql
# import jieba
# # str4=['1','2']
# # str1='sad'
# # str2='sad'
# # str3=str1+str2
# # str4.append(str3)
# # print(str4)
# str1=['玩的开始能否是你李开复南方都是你放开那倒是可拿到手','大萨达撒a']
# str3=[]
# # for i in range(len(str1)):
# #     str2=jieba.cut(str1[i])
# #     str3.append(str2)
# # def fenci_num(sentence):
# #     k=0
# #     for word in sentence:
# #         k=k+1
# #     return k
# # #print(' '.join(str3[0]))
# # print(fenci_num(str3[0]))
# str3.append(1)
# str3.append(2)
# print(str3)
# from keras.preprocessing import sequence
# from sklearn.model_selection import train_test_split
# import collections
# import nltk#用来分词
# import numpy as np#用来统计词频
# from keras.layers.core import Activation, Dense
# from keras.layers.embeddings import Embedding
# from keras.layers.recurrent import LSTM
# from keras.models import Sequential
# from keras.preprocessing import sequence
# from sklearn.model_selection import train_test_split
# import collections
# import nltk#用来分词
# import numpy as np#用来统计词频
# X=[[1,2],[3,4],[2,4]]
# y=[1,2,1]
# Xtrain, Xtest, ytrain, ytest =\
#     train_test_split(X, y, test_size=0.2, random_state=42)
#                         #样本比例，如果是整数的话就是样本的数量
# ## 网络构建
# EMBEDDING_SIZE = 128
# HIDDEN_LAYER_SIZE = 64
# BATCH_SIZE = 32
# NUM_EPOCHS = 10
# '''
# 这里损失函数用 binary_crossentropy， 优化方法用 adam。
#  至于 EMBEDDING_SIZE , HIDDEN_LAYER_SIZE ,
#   以及训练时用到的BATCH_SIZE 和 NUM_EPOCHS 这些超参数，
#   就凭经验多跑几次调优了。
# '''
# '''调试分割线'''
# print('3',Xtrain)
# print('4',ytrain)
# model = Sequential()
# #Embedding层只能作为模型的第一层
# #                  一共多少单词2000   第一层节点数128
# model.add(Embedding(5, 10
#                     #      每个句子长度40
#                     ,input_length=2))
# #                64            dropout将会减少这种过拟合
# model.add(LSTM(5, dropout=0.2
# #dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例
#                , recurrent_dropout=0.2))
# #recurrent_dropout：0~1之间的浮点数，
# #控制循环状态的线性变换的神经元断开比例
# model.add(Dense(1)) #全链接层，1代表该层的输出维度
# #激活函数未指定，若即使用线性激活函数：a(x)=x
# model.add(Activation("sigmoid")) #激活函数
# model.compile(loss="binary_crossentropy"
#               , optimizer="adam",metrics=["accuracy"])
# #                                             性能评估
# ## 网络训练
# # BATCH_SIZE = 32
# # NUM_EPOCHS = 10
# model.fit(Xtrain, ytrain, nb_epoch=500, batch_size=1,validation_data=(Xtest, ytest))
#
from requests import get
res = get('http://192.168.210.184:9200').text
print(res)


























