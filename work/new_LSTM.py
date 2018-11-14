#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : new_LSTM.py
# @Author: WangYe
# @Date  : 2018/8/6
# @Software: PyCharm
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import numpy as np#用来统计词频
import jieba
import os
'''读取文件'''
def write_file():
    path='.\\training.xls'
    question=[]
    answer=[]
    label=[]
    #pathlist=os.listdir(path)
    sentence=[]
    f=open(path,'r',encoding='utf-8')
    for i  in f.readlines(500000):
        #print(i)
        temp=list(i.split('\t'))
        question=temp[0]
        answer=temp[1]
        temp_sentence=question+answer
        sentence.append(temp_sentence)
        label.append(temp[2])
    #print(len(label))
    return sentence,label
'''分词'''
def fenci():
    sentece=write_file()[0]
    #print(len(sentece))
    label=write_file()[1]
    #print(len(label))
    label1=[]
    fenci_sentence = []
    for i in range(len(label)):   #标签处理
        label1.append(label[i].strip('\n'))
        temp_sentence=sentece[i].replace('，','').\
            replace('。','').replace('？','').\
            replace('、','').replace(' ','')
        fenci_sentence.append(jieba.lcut(temp_sentence))
    # b= ' '.join(fenci_sentence[0])
    # print(b)
    #print(' '.join(fenci_sentence[500]))
    # print('s',label1)
    return fenci_sentence,label1
'''构建词频矩阵'''
def WORD_freqs():
    sentence=fenci()[0]
    add_sentence_freqs=collections.Counter()
    for i in range(len(sentence)):
        for words in sentence[i]:
            add_sentence_freqs[words]+=1
    #print(add_sentence_freqs)
    return add_sentence_freqs
'''找分词之后长度的函数'''
def length_num(k):
    i=0
    for m in k:
        i=i+1
    return i

'''转换X，y训练集'''
def train_Xy():
    label=fenci()[1]
    word_num=len(WORD_freqs())
    sentence=fenci()[0]
    #print('dsa',type(sentence))
    # for i in range(2):
    #     print('+++', ' '.join(sentence[0]))
    # print('+++',' '.join(sentence[0]))
    # print('dsa',' '.join(sentence[0]))

    num_sentence=len(sentence)
    max_length = 0  #获取单句最大长度
    for i in range(len(sentence)):
        temp_length = length_num(sentence[i])
        if temp_length > max_length:
            max_length = temp_length
    # print('max_length:',max_length)
    word_freqs=WORD_freqs()
    # print('word_freqs:',word_freqs)
    # print('len:',len(word_freqs))
    seq_word= {x[0]: i + 2
                  for i, x in enumerate
                  (word_freqs.most_common(word_num))}
    # print('seq_word:',seq_word)
    # print('len:',len(seq_word))
    '''写入训练集'''
    X = np.empty(num_sentence, dtype=list)  # num_recs：样本数
    #      zeros(shape, dtype=None, order='C')
    #print('1', X)
    y = np.zeros(num_sentence)
    seq_word["PAD"] = 0  # 填充长度单词
    seq_word["UNK"] = 1  # 伪单词用1代替
    '''将X处理成 最大单句长度*所有句子数量 的矩阵'''
    for i in range(num_sentence):
        #print('i',i)
        temp=[]
        #print(' '.join(fenci()[0][i]))
        for word in sentence[i]:
            #print('word:',word)
            if word in seq_word:
                temp.append(seq_word[word])
            else:
                temp.append(seq_word["UNK"])
        X[i]=temp
    for i in range(len(label)):
        y[i]=int(label[i])
    # print('X:',X)
    # print('len',len(X))
    # print('shape',X.shape)
    # print('y',y)
    # print('len',len(y))
    X = sequence.pad_sequences(X, maxlen=max_length)
    return X,y,max_length,num_sentence,word_num

def deeplearning():
    X=train_Xy()[0]
    y=train_Xy()[1]
    max_length=train_Xy()[2]
    num_sentence=train_Xy()[3]
    word_num=train_Xy()[4]+2#伪单词和填充单词
    #print(max_length)
    Xtrain, Xtest, ytrain, ytest = \
                train_test_split\
                    (X, y, test_size=0.8, random_state=42)


    # 样本比例，如果是整数的话就是样本的数量
    model = Sequential()
    # Embedding层只能作为模型的第一层
    #                  一共多少单词 第一层节点数
    model.add(Embedding(word_num, 128,
                        input_length=max_length))
    #                     #      每个句子长度40
    #                64            dropout将会减少这种过拟合
    model.add(LSTM(64, dropout=0.2
                   # dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例
                   , recurrent_dropout=0.2))
    # recurrent_dropout：0~1之间的浮点数，
    # 控制循环状态的线性变换的神经元断开比例
    model.add(Dense(1))  # 全链接层，1代表该层的输出维度
    # 激活函数未指定，若即使用线性激活函数：a(x)=x
    model.add(Activation("sigmoid"))  # 激活函数
    model.compile(loss="binary_crossentropy"
                  , optimizer="adam", metrics=["accuracy"])
    #                                             性能评估
    ## 网络训练
    # BATCH_SIZE = 32
    # NUM_EPOCHS = 10
    model.fit(Xtrain, ytrain, batch_size=50
              , epochs=10, validation_data=(Xtest, ytest))
    score, acc = model.evaluate(Xtest, ytest,
                                batch_size=32)
    print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
    #print('{}   {}      {}'.format('预测', '真实', '句子'))
if __name__ == '__main__':
    #fenci()
    #WORD_freqs()
    #train_Xy()
    #write_file()
    deeplearning()