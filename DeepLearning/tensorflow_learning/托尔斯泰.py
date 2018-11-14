#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : 托尔斯泰.py
# @Author: WangYe
# @Date  : 2018/9/14
# @Software: PyCharm
import tensorflow as tf
# m1 = tf.constant([[3,3]])  #一行两列
# m2 = tf.constant([[2],[3]])#两行一列
# #相乘
# product = tf.matmul(m1,m2)
# #print(product)
# #Tensor("MatMul:0", shape=(1, 1), dtype=int32)
#
#
# #启动会话
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()
'''变量测试'''
# x = tf.Variable([1,2])#变量
# a = tf.constant([3,3])#常量
# #增加一个减法op
# sub = tf.subtract(x,a)
# #增加一个加法op
# add = tf.add(x,sub)
#
# #变量初始化
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(sub))
#     print(sess.run(add))

#0创建一个变量，初始化为0
# state = tf.Variable(0,name='counter')
# new_value = tf.add(state,1)
# # tf中的赋值方法,类似state==new_value
# update = tf.assign(state,new_value)
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(state))
#     for _ in range(5):
#         sess.run(update)
#         print(sess.run(state))

'''fetch and feed'''
#Fetch：在绘画中可以执行多个op，预存结果
#
# input1 = tf.constant(3.0)
# input2 = tf.constant(2.0)
# input3 = tf.constant(5.0)
#
# add = tf.add(input2,input3)
# mul = tf.multiply(input1,add)
#
# with tf.Session() as sess:
#     result = sess.run([mul,add])
#     print(result)

#feed
#创建占位符
# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)
# output = tf.multiply(input1,input2)
# with tf.Session() as sess:
#     print(sess.run(output,feed_dict={input1:[8,3],input2:[2,4]}))
'''案例'''
import numpy as np

x_date = np.random.rand(100)
y_date = x_date*0.1 + 0.2

#创建一个线性模型
#不断改变 b 和 k的值来优化
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_date + b

#二次代价函数
#           求均值          平方
loss = tf.reduce_mean(tf.square(y_date-y))
#定义一个梯度下降模型
#                           学习率
optimizer = tf.train.GradientDescentOptimizer(0.2)

#最小化代价函数

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):  #迭代
        sess.run(train)
        if step%20 == 0:
            print(step,sess.run([k,b]))