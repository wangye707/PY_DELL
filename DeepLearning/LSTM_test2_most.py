#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : LSTM_test2_most.py
# @Author: WangYe
# @Date  : 2018/8/2
# @Software: PyCharm
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import nltk#用来分词
import numpy as np#用来统计词频

## EDA
maxlen = 0  #句子最大长度
word_freqs = collections.Counter()  #词频
num_recs = 0   #样本数
with open('C:/Users/wy/Desktop/data/RNN_LSTM/train_data.txt','r+'
        ,encoding='utf-8') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1  #样本数叠加/
print(num_recs)#7086个样本
print(label)
print('max_len ',maxlen)
print('nb_words ', len(word_freqs))
print('sad',word_freqs)
'''
可见一共有 2324 个不同的单词，包括标点符号。每句话最多包含 42 个单词。 
根据不同单词的个数 (nb_words)，我们可以把词汇表的大小设为一个定值，
并且对于不在词汇表里的单词，把它们用伪单词 UNK 代替。 
根据句子的最大长度 (max_lens)，我们可以统一句子的长度
，把短句用 0 填充。 依前所述，我们把VOCABULARY_SIZE 设为 2002。
包含训练数据中按词频从大到小排序后的前 2000 个单词
，外加一个伪单词 UNK 和填充单词 0。
最大句子长度 MAX_SENTENCE_LENGTH 设为40。
'''
## 准备数据
MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40
#接下来建立两个 lookup tables，分别是 word2index 和
#index2word，用于单词和数字转换。
#                             词频 ：外加一个伪单词 UNK 和填充单词 0
#print(len(word_freqs))
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i+2
              for i, x in enumerate
              (word_freqs.most_common(MAX_FEATURES))}
print('+++++++++++++++', word2index, '+++++++++++++++')
#                         most_common([n])函数
#返回一个TopN列表。如果n没有被指定，则返回所有元素。
#当多个元素计数值相同时，排列是无确定顺序的。
word2index["PAD"] = 0  #填充长度单词
word2index["UNK"] = 1  #伪单词用1代替
index2word = {v:k for k, v in word2index.items()}
#                       item()返回所有健值的列表
#      empty(shape, dtype=None, order='C')
X = np.empty(num_recs,dtype=list)#num_recs：样本数
#      zeros(shape, dtype=None, order='C')
print('1',X)
y = np.zeros(num_recs)
print('2',y)
i=0
with open('C:/Users/wy/Desktop/data/RNN_LSTM/train_data.txt'
        ,'r+',encoding='utf-8') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])

            else:
                seqs.append(word2index["UNK"])
        #写入数据和标签
        X[i] = seqs
        y[i] = int(label)
        i += 1
# print('*********', X, '*********')
#转换长度，形成矩阵，让长度统一
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
#print(X)
## 数据划分
print(X.shape)
print(y.shape)
Xtrain, Xtest, ytrain, ytest =\
    train_test_split(X, y, test_size=0.2, random_state=42)
                        #样本比例，如果是整数的话就是样本的数量
## 网络构建
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 10
'''
这里损失函数用 binary_crossentropy， 优化方法用 adam。
 至于 EMBEDDING_SIZE , HIDDEN_LAYER_SIZE ,
  以及训练时用到的BATCH_SIZE 和 NUM_EPOCHS 这些超参数，
  就凭经验多跑几次调优了。 
'''
'''调试分割线'''
print('3',Xtrain)
print('4',ytrain)
model = Sequential()
#Embedding层只能作为模型的第一层
#                  一共多少单词2000   第一层节点数128
model.add(Embedding(vocab_size, EMBEDDING_SIZE
                    #      每个句子长度40
                    ,input_length=MAX_SENTENCE_LENGTH))
#                64            dropout将会减少这种过拟合
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2
#dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例
               , recurrent_dropout=0.2))
#recurrent_dropout：0~1之间的浮点数，
#控制循环状态的线性变换的神经元断开比例
model.add(Dense(1)) #全链接层，1代表该层的输出维度
#激活函数未指定，若即使用线性激活函数：a(x)=x
model.add(Activation("sigmoid")) #激活函数
model.compile(loss="binary_crossentropy"
              , optimizer="adam",metrics=["accuracy"])
#                                             性能评估
## 网络训练
# BATCH_SIZE = 32
# NUM_EPOCHS = 10
model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE
          , epochs=NUM_EPOCHS,validation_data=(Xtest, ytest))
# '''调试分割线'''
# 预测
score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
print('{}   {}      {}'.format('预测','真实','句子'))
for i in range(5):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1,40)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for
                     x in xtest[0] if x != 0])
    print(' {}      {}     {}'.format
          (int(round(ypred)), int(ylabel), sent))

from keras.layers.core import VectorAssembler
# ##### 自己输入
# INPUT_SENTENCES = ['I love reading.','You are so boring.']
# XX = np.empty(len(INPUT_SENTENCES),dtype=list)
# i=0
# for sentence in  INPUT_SENTENCES:
#     words = nltk.word_tokenize(sentence.lower())
#     seq = []
#     for word in words:
#         if word in word2index:
#             seq.append(word2index[word])
#         else:
#             seq.append(word2index['UNK'])
#     XX[i] = seq
#     i+=1
#
# XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
# labels = [int(round(x[0])) for x in model.predict(XX) ]
# label2word = {1:'积极', 0:'消极'}
# for i in range(len(INPUT_SENTENCES)):
#     print('{}   {}'.format(
#         label2word[labels[i]], INPUT_SENTENCES[i]))