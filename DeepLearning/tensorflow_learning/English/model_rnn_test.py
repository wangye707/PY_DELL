#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : model_rnn.py
# @Author: WangYe
# @Date  : 2018/11/2
# @Software: PyCharm
import tensorflow as tf
from sklearn.model_selection import train_test_split
import collections
import nltk#用来分词
import numpy as np#用来统计词频
import time
with open('./English_X.txt','r',encoding='utf-8') as f:

    words = f.read().split(',')
    temp_xlist=[]
    for word in words:
        #print(type(word))
        try:
            temp_xlist.append(int(word))
        except ValueError:
            pass
    X=temp_xlist
    f.close()
with open('./English_y.txt','r',encoding='utf-8') as f1:
    words = f1.read().replace(' ','').split(',')
    temp_ylist = []
    #print(len(words))
    for word in words:
        #print(float(word))
        temp_ylist.append(int(float(word)))
    y = temp_ylist
    f1.close()
num_recs=7086  #样本单词数量
y_temp=[]
print(len(X))
print(len(y))
for i in range(len(y)):
    if y[i]==0:
        temp_ylist=[1,0]
    else:
        temp_ylist=[0,1]
    y_temp.append(temp_ylist)
# print(len(X))
# print(len(y))


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

learning_rate = 0.001
train_step = 1000
batch_size = 50
display_step = 100
start=time.time()

frame_size = 41
# frame_size: 序列里面每一个分量的单词数量
sequence_length = 7086
# sequence_size: 每个样本序列的长度
hidden_num = 128
n_classes = 2
x = tf.placeholder(dtype=tf.float32, shape=[None,41], name="inputx")
y = tf.placeholder(dtype=tf.float32, shape=[None,2], name="expected_y")
weights = tf.Variable(tf.truncated_normal(shape=[hidden_num, n_classes],stddev=0.1))
bias = tf.Variable(tf.constant(0.1,shape=[n_classes]))


def loss1(label, pred):
    return tf.square(label - pred)


def get_batches(x, y, batch_size=100):
    n_batches = len(x) // batch_size
    #  // 指 取整除 - 返回商的整数部分（向下取整
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]

    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]

def RNN(x,weights,bias):
    x=tf.reshape(x,shape=[-1,41,1])
    # print(type(x))
    rnn_cell=tf.nn.rnn_cell.BasicRNNCell(hidden_num)
    init_state=tf.zeros(shape=[batch_size,rnn_cell.state_size])
    # 其实这是一个深度RNN网络,对于每一个长度为n的序列[x1,x2,x3,...,xn]的每一个xi,都会在深度方向跑一遍RNN,跑上hidden_num个隐层单元
    output,states=tf.nn.dynamic_rnn(rnn_cell,x,dtype=tf.float32)
    #print(output[1].shpae)
    return tf.nn.softmax(tf.matmul(output[:,-1],weights)+bias,1)
predy=RNN(x,weights,bias)
# cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predy,labels=y))
# train=tf.train.AdamOptimizer(train_rate).minimize(cost)
loss_value = loss1(y, predy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

correct_pred=tf.equal(tf.argmax(predy,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.to_float(correct_pred))

grads_and_vars = optimizer.compute_gradients(loss_value)
#按batch_size划分数据集
#Xtrains,ytrains=tf.train.batch([Xtrain,ytrain],batch_size=batch_size)
# del
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    step=10
    testx,testy=Xtest,ytest
    num=0
    while step<train_step:
        num=num+1
        #start_x=step*batch_size%len(Xtrain)

        #batch_x,batch_y=Xtrains[num],ytrains[num]

        #batch_x=tf.reshape(batch_x,shape=[sequence_length,frame_size])
        for ii, (Xtrains, ytrains) in enumerate(get_batches(Xtrain, ytrain, batch_size), 1):
            feed = {x : Xtrains,
                   # y : ytrains[:,None]
                    y: ytrains

                    }
           # print("aaaaaaaaaaaaaaaaa")
            _loss,__=sess.run([loss_value,grads_and_vars],feed_dict=feed)
            if step % display_step ==0:

                acc,loss=sess.run([accuracy,loss_value],feed_dict={x:testx,y:testy})
                print('train:',step,acc,loss)

            if step % display_step ==0:

                acc,loss=sess.run([accuracy,loss_value],feed_dict={x:testx,y:testy})
                print('test:',step,acc,loss)

            step+=1
end=time.time()
print("costing time:",end-start)