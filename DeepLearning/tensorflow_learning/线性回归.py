#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : 线性回归.py
# @Author: WangYe
# @Date  : 2018/9/14
# @Software: PyCharm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#          随机生成                 生成二维度的
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]

noise = np.random.normal(0,0.02,x_data.shape)

y_data = np.square(x_data) + noise

#                            任意行  一列
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#定义神经网络
#                           输入为1，中间为10
Weight_L1 = tf.Variable(tf.random_normal([1,10]))
#初始化为0
biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,Weight_L1) + biases_L1
#激活函数
L1 = tf.nn.tanh(Wx_plus_b_L1)


#定义输出层
#                               中间为10，输出为1
Weight_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = biases_L1 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weight_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)  #预测值尚未

#二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    prediction_value = sess.run(prediction,feed_dict={x:x_data})

    plt.figure()
    plt.scatter(x_data,y_data)
    #                       红色实线  宽度
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()