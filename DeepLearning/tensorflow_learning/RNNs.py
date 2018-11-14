#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : RNNs.py
# @Author: WangYe
# @Date  : 2018/10/31
# @Software: PyCharm
#coding:utf-8
from __future__ import print_function
import os, time
from PIL import Image
import numpy
import random
import  tensorflow as tf
from  tensorflow.examples.tutorials.mnist import  input_data
from tensorflow.contrib import  rnn

mnist=input_data.read_data_sets(r"./wy_data",one_hot=True)

train_rate=0.001
train_step=10000
batch_size=1280
display_step=100

frame_size=28
sequence_length=28
hidden_num=100
n_classes=10

#定义输入,输出
x=tf.placeholder(dtype=tf.float32,shape=[None,sequence_length*frame_size],name="inputx")
y=tf.placeholder(dtype=tf.float32,shape=[None,n_classes],name="expected_y")
#定义权值
weights=tf.Variable(tf.truncated_normal(shape=[hidden_num,n_classes]))
bias=tf.Variable(tf.zeros(shape=[n_classes]))

def RNN(x,weights,bias):
    x=tf.reshape(x,shape=[-1,sequence_length,frame_size])
    rnn_cell=tf.nn.rnn_cell.BasicRNNCell(hidden_num)
    init_state=tf.zeros(shape=[batch_size,rnn_cell.state_size])
    # 其实这是一个深度RNN网络,对于每一个长度为n的序列[x1,x2,x3,...,xn]的每一个xi,都会在深度方向跑一遍RNN,跑上hidden_num个隐层单元
    output,states=tf.nn.dynamic_rnn(rnn_cell,x,dtype=tf.float32)
    return tf.nn.softmax(tf.matmul(output[:,-1,:],weights)+bias,1)
predy=RNN(x,weights,bias)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predy,labels=y))
train=tf.train.AdamOptimizer(train_rate).minimize(cost)

correct_pred=tf.equal(tf.argmax(predy,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.to_float(correct_pred))

sess=tf.Session()
sess.run(tf.initialize_all_variables())
step=1
testx,testy=mnist.test.next_batch(batch_size)
while step<train_step:
    batch_x,batch_y=mnist.train.next_batch(batch_size)
#    batch_x=tf.reshape(batch_x,shape=[batch_size,sequence_length,frame_size])
    _loss,__=sess.run([cost,train],feed_dict={x:batch_x,y:batch_y})
    if step % display_step ==0:

        acc,loss=sess.run([accuracy,cost],feed_dict={x:testx,y:testy})
        print(step,acc,loss)

    step+=1