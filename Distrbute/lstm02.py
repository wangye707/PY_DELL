# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:49:44 2017

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:40:09 2017

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:17:58 2017

@author: ultra
"""

import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词
 
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from sklearn.metrics import accuracy_score

from keras.layers import Embedding, Flatten, Input, merge
# 
##from __future__ import absolute_import #导入3.x的特征函数
##from __future__ import print_function
 
#neg=pd.read_excel('neg.xls',header=None,index=None)
#pos=pd.read_excel('pos.xls',header=None,index=None) #读取训练语料完毕
#pos['mark']=1
#neg['mark']=0 #给训练语料贴上标签
#pn=pd.concat([pos,neg],ignore_index=True) #合并语料
#neglen=len(neg)
#poslen=len(pos) #计算语料数目
# 
#cw = lambda x: list(jieba.cut(x)) #定义分词函数
#pn['words'] = pn[0].apply(cw)
#
#comment = pd.read_excel('sum.xls') #读入评论内容
##comment = pd.read_csv('a.csv', encoding='utf-8')
#comment = comment[comment['rateContent'].notnull()] #仅读取非空评论
#comment['words'] = comment['rateContent'].apply(cw) #评论分词 
# 
#d2v_train = pd.concat([pn['words'], comment['words']], ignore_index = True) 
#print(pn[:4]) 
#w = [] #将所有词语整合在一起, zhu:费时间
#for i in d2v_train:
#  w.extend(i)
#
#dict = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数
#dict['id']=list(range(1,len(dict)+1))
#print(pn[:4]) 
#get_sent = lambda x: list(dict['id'][x])
#pn['sent'] = pn['words'].apply(get_sent) #速度太慢
#print(pn[:4]) 
#maxlen = 50
# 
#print("Pad sequences (samples x time)")
#pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))
# 
#x = np.array(list(pn['sent']))[::2] #训练集
#y = np.array(list(pn['mark']))[::2]
#xt = np.array(list(pn['sent']))[1::2] #测试集
#yt = np.array(list(pn['mark']))[1::2]
#xa = np.array(list(pn['sent'])) #全集
#ya = np.array(list(pn['mark']))
#
#import pickle
#pickle.dump((x,y,xt,yt,xa,ya,maxlen,dict,comment),open('data.pkl','wb'))
x,y,xt,yt,xa,ya,maxlen,dict,comment = pickle.load(open('data.pkl','rb'))



 
print('Build model...')
 
sent = Input((50, ), name='input') 
 
latent_dim = 256
sent_embedding = Embedding( len(dict)+1, latent_dim, name='word_embedding', input_length=50)(sent)
sent_encoded = LSTM(128,return_sequences=True)(sent_embedding)#return_sequences=True
sent_encoded = LSTM(128)(sent_encoded)
#sent_encoded = Dropout(0.5)(sent_encoded)
sent_encoded = Dense(1)(sent_encoded)
loss = Activation('sigmoid')(sent_encoded)

model = Model(inputs= sent,outputs=loss)
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(xa, ya, batch_size=128, epochs=2)

classes = model.predict(xa)
pred=[]
for c in classes:
    if c>=0.5:
        pred.append(1)
    else:
        pred.append(0)
pred = np.array(pred)
        
acc = accuracy_score(ya,pred)
 

print('Test accuracy:', acc)

