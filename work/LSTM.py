#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : LSTM.py
# @Author: WangYe
# @Date  : 2018/8/3
# @Software: PyCharm
#第一次构建自己的神经网络
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import collections
#import nltk#用来分词
import numpy as np#用来统计词频
import jieba
import os
'''读取文件'''
def write_file():
    path='C:\\Users\\wy\\Desktop\\data\\' \
         'RNN_LSTM\\training.xls'
    question=[]
    answer=[]
    label=[]
    #pathlist=os.listdir(path)
    f=open(path,'r',encoding='utf-8')
    for i  in f.readlines():
        temp=list(i.split('\t'))
        question.append(temp[0])
        answer.append(temp[1])
        label.append(temp[2])
    #print(len(label))
    return question,answer,label
'''分词'''
def fenci():
    question=write_file()[0]

    answer=write_file()[1]

    label=write_file()[2]

    label1=[]
    right_sentence = []
    wrong_sentence = []
    for i in range(len(label)):   #标签处理
        label1.append(label[i].strip('\n'))

    '''将0，1的问题答案分类'''
    for i in range(len(label1)):
        #print(i)
        if label1[i] == '1':
            #print(i)
            right_temp = jieba.cut(question[i] + answer[i])
            #print(right_temp)
            #print(' '.join(right_temp))
            right_sentence.append(right_temp)
        if label1[i] == '0':
            wrong_temp = jieba.cut(question[i] + answer[i])

            wrong_sentence.append(wrong_temp)
    #print(' '.join(wrong_sentence[8]))
    return right_sentence,wrong_sentence,label1
'''构建词频矩阵'''
def WORD_freqs():
    right_sentence=fenci()[0]
    #print(' '.join(right_sentence[5]))
    wrong_sentence=fenci()[1]
    #print(' '.join(wrong_sentence[800]))
    add_sentence=right_sentence+wrong_sentence
    # right_sentence_freqs = collections.Counter()
    # wrong_sentence_freqs = collections.Counter()
    add_sentence_freqs=collections.Counter()
    #print(len(right_sentence))  #长度为155102
    # print(len(right_sentence))
    # print(len(wrong_sentence))
    for i in range(len(add_sentence)):
        for words in add_sentence[i]:
            add_sentence_freqs[words]+=1
    # for i in range(len(wrong_sentence)):
    #     for words in wrong_sentence[i]:
    #         wrong_sentence_freqs[words]+=1
    #print(right_sentence_freqs)
    #print(' '.join(right_sentence[i]))
    return add_sentence_freqs
'''统计分词之后的单词数量'''
def fenci_num(sentence):
    k=0
    for word in sentence:
        k=k+1
    return k

'''计算单词样本数，单句最大长度，所有句子数量'''
def numbers():
    '''单词样本数：'''
    word_num=len(WORD_freqs())
    #print(word_num)
    '''单句最大长度：'''
    right_sentence=fenci()[0]
    #print(fenci_num(right_sentence[0]))
    wrong_sentence=fenci()[1]
    right_sentence_max_length=0
    wrong_sentence_max_length=0
    for i in range(len(right_sentence)):
        temp=fenci_num(right_sentence[i])
        if temp>right_sentence_max_length:
            #temp=0
            #print(fenci_num(right_sentence[i]))
            right_sentence_max_length=temp
            #print('q',len(right_sentence[i]))
            #print('qlength',i)
    for i in range(len(wrong_sentence)):
        temp=fenci_num(wrong_sentence)
        if temp>wrong_sentence_max_length:
            wrong_sentence_max_length=temp
            #print('a',len(wrong_sentence[i]))
            #print('length',i)
    #print('dsa',answer[49417])
    '''所有句子数量'''
    sentence_num1=len(right_sentence)
    sentence_num2=len(wrong_sentence)
    sentence_num=sentence_num1+sentence_num2
    '''标签处理'''
    label = []
    for i in range(sentence_num):
        if i < len(right_sentence):
            label.append(1)
        else:
            label.append(0)
    #输出测试
    print('所有单词数量',word_num)
    print('正确句子数量',len(right_sentence))
    print('错误句子数量',len(wrong_sentence))
    print('标签数量',len(label))
    print('正确句子最大长度',right_sentence_max_length)
    print('错误句子最大长度',wrong_sentence_max_length)
    print('所有句子数量',sentence_num)
    return word_num,label,sentence_num,len(wrong_sentence)

'''
所有单词数量 218232
正确句子数量 155102
错误句子数量 8286
标签数量 163388
正确句子最大长度 2087
错误句子最大长度 8286
所有句子数量 163388
'''
'''单词矩阵转换数字矩阵,并且补齐单个句子'''
def transfrom_word_int():
    vectorizer = CountVectorizer()
    temp_sentence=[]
    all_sentence=[]
    word_num=numbers()[0]
    max_length=numbers()[3]
    all_sentence=fenci()[0]+fenci()[1]#正确+错误的所有词频集合
    # print(all_sentence[0])
    # for i in range(len(all_sentence)):
    #     temp=' '.join('%s' %id for id in all_sentence[i])
    #     temp_sentence.append(temp)
    #print(' ',temp_sentence[1])
    temp_freqs=WORD_freqs()
    print(WORD_freqs())
    #num_arrary=vectorizer.fit_transform(all_sentence)
    #print(num_arrary.shape())
    #return num_arraryX = np.empty(num_recs,dtype=list)#num_recs：样本数
#      zeros(shape, dtype=None, order='C')
    X = np.empty(word_num, dtype=list)  # num_recs：样本数
    print('X1',X)
    y = np.zeros(word_num)
    print('y1',y)
    seqs=[]
    for num in range(len(all_sentence)):
        for word in all_sentence[num]:
            if word in temp_freqs:
                seqs.append(temp_freqs[word])
            else:
                seqs.append(temp_freqs["UNK"])
                #写入数据和标签
        X[num] = seqs
#print(X)
#转换长度，形成矩阵，让长度统一
    #X = sequence.pad_sequences(X, maxlen=max_length)
    print(X)
    return X
# def deep_learning():
#     X=transfrom_word_int()  #训练集X，y
#     print(X.shape())
#     y=numbers()[1]
#     print(y.shape())
#     Xtrain, Xtest, ytrain, ytest = \
#         train_test_split(X, y, test_size=0.8, random_state=42)
#     # 样本比例，如果是整数的话就是样本的数量
#     model = Sequential()
#     # Embedding层只能作为模型的第一层
#     #                  一共多少单词2000   第一层节点数128
#     model.add(Embedding(218232, 128
#                         #      每个句子长度40
#                         , input_length=8286))
#     #                64            dropout将会减少这种过拟合
#     model.add(LSTM(64, dropout=0.2
#                    # dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例
#                    , recurrent_dropout=0.2))
#     # recurrent_dropout：0~1之间的浮点数，
#     # 控制循环状态的线性变换的神经元断开比例
#     model.add(Dense(1))  # 全链接层，1代表该层的输出维度
#     # 激活函数未指定，若即使用线性激活函数：a(x)=x
#     model.add(Activation("sigmoid"))  # 激活函数
#     model.compile(loss="binary_crossentropy"
#                   , optimizer="adam", metrics=["accuracy"])
#     #                                             性能评估
#     ## 网络训练
#     # BATCH_SIZE = 32
#     # NUM_EPOCHS = 10
#     model.fit(Xtrain, ytrain, batch_size=32
#               , epochs=10, validation_data=(Xtest, ytest))
#     score, acc = model.evaluate(Xtest, ytest,
#                                 batch_size=32)
#     print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
#     print('{}   {}      {}'.format('预测', '真实', '句子'))
if __name__ == '__main__':
    #write_file()
    #numbers()
    #fenci()
    #WORD_freqs()
    transfrom_word_int()
    #deep_learning()