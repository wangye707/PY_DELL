#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : english_predict.py
# @Author: WangYe
# @Date  : 2018/11/1
# @Software: PyCharm
#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : LSTM_test2_most.py
# @Author: WangYe
# @Date  : 2018/8/2
# @Software: PyCharm
from tensorflow.contrib import  rnn
#import tensorflow as tf
from sklearn.model_selection import train_test_split
import collections
import nltk#用来分词
import numpy as np#用来统计词频
## EDA
maxlen = 0  #句子最大长度
label_list=[]
word_freqs = collections.Counter()  #词频
num_recs = 0   #样本数
with open('./train_data.txt','r+'
        ,encoding='utf-8') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
        label_list.append(label)
        words = nltk.word_tokenize(sentence.lower())
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1  #样本数叠加/
    f.close()
# print(num_recs)#7086个样本
# print(label_list)  #标签列表
# print('max_len ',maxlen) #最大长度42
# print('nb_words ', len(word_freqs))  #一共2330个单词
# print('sad',word_freqs) #词频字典

## 准备数据
MAX_FEATURES = 2000  #取前2000个单词
MAX_SENTENCE_LENGTH = 40  #单句最大长度40个单词
#接下来建立两个 lookup tables，分别是 word2index 和
#index2word，用于单词和数字转换。
#                             词频 ：外加一个伪单词 UNK 和填充单词 0
#print(len(word_freqs))
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2  #单词总数   2是伪单词和填充单词0
word2index = {x[0]: i+2 for i, x in enumerate (word_freqs.most_common(MAX_FEATURES))}
#print('+++++++++++++++', word2index, '+++++++++++++++')
#most_common([n])函数
#返回一个TopN列表。如果n没有被指定，则返回所有元素。
#当多个元素计数值相同时，排列是无确定顺序的。
word2index["PAD"] = 0  #填充长度单词
word2index["UNK"] = 1  #伪单词用1代替
index2word = {v:k for k, v in word2index.items()}
#                       item()返回所有健值的列表
#      empty(shape, dtype=None, order='C')
#X = np.empty(num_recs,dtype=list)#num_recs：样本数
#      zeros(shape, dtype=None, order='C')
#print('1',X)  #7086行
X = []
y = np.zeros(num_recs)
#print('2',y) #7086行
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
                #不认识的单词填充
                seqs.append(word2index["UNK"])
        #转换长度，形成矩阵，让长度统一
        if len(seqs) < MAX_SENTENCE_LENGTH:
            difference = MAX_SENTENCE_LENGTH - len(seqs)
            for temp in range(difference + 1):
                seqs.append(word2index["PAD"])
        #print(len(seqs))
        #写入数据和标签
        X=X+seqs
        y[i] = int(label)
        i += 1
    f.close()
    '''处理y为标签矩阵'''
#print(len(X))
print(y)
with open('./English_X.txt','w',encoding='utf-8') as f:
    for i in X:
        f.write(',')
        f.write(str(i))

    f.close()
with open('./English_y.txt','w',encoding='utf-8') as f:
    # for i in y:
    for i in y:
        f.write(',')
        f.write(str(i))
    f.close()
y_temp=[]
for i in range(len(y)):
    if y[i]==0:
        temp_ylist=[1,0]
    else:
        temp_ylist=[0,1]
    y_temp.append(temp_ylist)
X=np.array(X).reshape(num_recs,41)
y=np.array(y_temp).reshape(num_recs,2)
        #X为 7086*41的矩阵
#划分训练集和测试集
# X=np.array(X,(41,7086))
# print(X.shape)
# print(y.shape)
Xtrain, Xtest, ytrain, ytest =\
    train_test_split(X, y, test_size=0.2, random_state=42)
print(Xtrain.shape)
print(ytrain.shape)
print(Xtest.shape)
print(ytest.shape)

#
#
# '''
# 可见一共有 2324 个不同的单词，包括标点符号。每句话最多包含 42 个单词。
# 根据不同单词的个数 (nb_words)，我们可以把词汇表的大小设为一个定值，
# 并且对于不在词汇表里的单词，把它们用伪单词 UNK 代替。
# 根据句子的最大长度 (max_lens)，我们可以统一句子的长度
# ，把短句用 0 填充。 依前所述，我们把VOCABULARY_SIZE 设为 2002。
# 包含训练数据中按词频从大到小排序后的前 2000 个单词
# ，外加一个伪单词 UNK 和填充单词 0。
# 最大句子长度 MAX_SENTENCE_LENGTH 设为41。
# '''
# train_rate = 0.001
# train_step = 1000
# batch_size = 50
# display_step = 100
#
# frame_size = vocab_size
# # frame_size: 序列里面每一个分量的大小
# sequence_length = 7086
# # sequence_size: 每个样本序列的长度
# hidden_num = 100
# n_classes = 2
# x = tf.placeholder(dtype=tf.float32, shape=[None,41], name="inputx")
# y = tf.placeholder(dtype=tf.float32, shape=[None,2], name="expected_y")
#
# weights = tf.Variable(tf.truncated_normal(shape=[hidden_num, n_classes],stddev=0.1))
# bias = tf.Variable(tf.constant(0.1,shape=[n_classes]))
#
# def RNN(x,weights,bias):
#     x=tf.reshape(x,shape=[-1,41,1])
#     # print(type(x))
#     rnn_cell=tf.nn.rnn_cell.BasicRNNCell(hidden_num)
#     init_state=tf.zeros(shape=[batch_size,rnn_cell.state_size])
#     # 其实这是一个深度RNN网络,对于每一个长度为n的序列[x1,x2,x3,...,xn]的每一个xi,都会在深度方向跑一遍RNN,跑上hidden_num个隐层单元
#     output,states=tf.nn.dynamic_rnn(rnn_cell,x,dtype=tf.float32)
#     #print(output[1].shpae)
#     return tf.nn.softmax(tf.matmul(output[:,-1],weights)+bias,1)
# predy=RNN(x,weights,bias)
# cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predy,labels=y))
# train=tf.train.AdamOptimizer(train_rate).minimize(cost)
#
# correct_pred=tf.equal(tf.argmax(predy,1),tf.argmax(y,1))
# accuracy=tf.reduce_mean(tf.to_float(correct_pred))
#
#
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     step=10
#     testx,testy=Xtest,ytest
#     while step<train_step:
#         #start=step*batch_size%len(Xtrain)
#         batch_x,batch_y=Xtrain,ytrain
#
#         #batch_x=tf.reshape(batch_x,shape=[sequence_length,frame_size])
#         _loss,__=sess.run([cost,train],feed_dict={x:batch_x,y:batch_y})
#
#         if step % display_step ==0:
#
#             acc,loss=sess.run([accuracy,cost],feed_dict={x:testx,y:testy})
#             print(step,acc,loss)
#
#         step+=1
