#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : tensoflow_test.py
# @Author: WangYe
# @Date  : 2018/7/25
# @Software: PyCharm
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random
# 参数
learning_rate = 0.01   #学习率
training_epochs = 1000  #迭代次数
display_step = 50   #每隔50次迭代，打印日志
# 训练数据
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]  #处理X的一个维度
X = tf.placeholder("float")  #占位符
Y = tf.placeholder("float")

# 模型参数
'''犹豫w和b会不断更新，所以用variable来保存'''
W = tf.Variable(rng.randn(), name="weight") #模型的偏重，最开始为随机的
b = tf.Variable(rng.randn(), name="bias")   #模型的权值

# 构建线性模型
pred = tf.add(tf.multiply(X, W), b)   #x*w+b=y

# 求误差(均方差)
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# 使用梯度下降拟合数据
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# 初始化所有变量（variables）
init = tf.initialize_all_variables()
# 开始训练
#先启动session
with tf.Session() as sess:
    sess.run(init)  #run初始化的variables

    # 灌入所有训练数据
    for epoch in range(training_epochs):  #遍历每一次的迭代次数
        for (x, y) in zip(train_X, train_Y):  #zip：将两个列表变成一个字典
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #Display logs per epoch step
        if (epoch+1) % display_step == 0:   #每隔50个
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print ("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print ("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    #Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()