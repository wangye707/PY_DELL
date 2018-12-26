#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : speech.py
# @Author: WangYe
# @Date  : 2018/12/19
# @Software: PyCharm
from __future__ import division, print_function, absolute_import
import tflearn
from sklearn.model_selection import train_test_split
import speech_data
from tensorflow.contrib import rnn
import tensorflow as tf

learning_rate = 0.001
train_step = 100000
batch_size = 128
display_step = 100
#start=time.time()

frame_size = 80
# frame_size: 序列里面每一个分量的单词数量

input_num = 20

#sequence_length = 7086
# sequence_size: 每个样本序列的长度
hidden_num = 128
n_classes = 10

batch = word_batch = speech_data.mfcc_batch_generator(batch_size)
X, Y = next(batch)
Xtrain, Xtest, ytrain, ytest =\
    train_test_split(X, Y, test_size=0.2, random_state=42)


x = tf.placeholder(dtype=tf.float32, shape=[None,input_num,frame_size], name="inputx")
y = tf.placeholder(dtype=tf.float32, shape=[None,n_classes], name="expected_y")
weights = tf.Variable(tf.truncated_normal(shape=[hidden_num, n_classes],stddev=0.1))
bias = tf.Variable(tf.constant(0.1,shape=[n_classes]))


def loss1(label, pred):
    return tf.square(label - pred)

def RNN(x,weights,bias):
    x=tf.reshape(x,shape=[-1,input_num,frame_size])
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
y_ = tf.nn.softmax(predy, name="loss_value")
loss_value = tf.reduce_mean((-1) * y_ * tf.log(tf.clip_by_value(y, 1e-4, 1.0)), name="loss_value")
        # y = tf.nn.softmax(out_layer, name="output")
        # loss_value = loss1(y, predy)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate,name="optimizer")
#loss_value = loss1(y, predy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

correct_pred=tf.equal(tf.argmax(predy,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.to_float(correct_pred))

grads_and_vars = optimizer.compute_gradients(loss_value)



def get_batches(x, y, batch_size=100):
    n_batches = len(x) // batch_size
    #  // 指 取整除 - 返回商的整数部分（向下取整
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]

    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    step=10
    testx,testy=Xtest,ytest
    while step<train_step:
        #start=step*batch_size%len(Xtrain)
        for ii, (Xtrains, ytrains) in enumerate(get_batches(Xtrain, ytrain, batch_size), 1):

            feed = {x: Xtrains,
                    # y : ytrains[:,None]
                    y: ytrains
                    }
            # print("aaaaaaaaaaaaaaaaa")
            _, loss_v, step = sess.run([optimizer, loss_value, train_step],
                                       feed_dict=feed)
        if step % display_step ==0:

            acc,loss=sess.run([accuracy,loss_value],feed_dict={x:testx,y:testy})
            print('train:',step,acc)

        if step % display_step ==0:

            acc,loss=sess.run([accuracy,loss_value],feed_dict={x:testx,y:testy})
            print('test:',step,acc)
        step = step + 1


