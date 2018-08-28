#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : image_test.py
# @Author: WangYe
# @Date  : 2018/7/31
# @Software: PyCharm
'''softmax回归测试MNIST数据集'''
import inputdata
import tensorflow as tf
#导入MNIST数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#softmax回归模型中的x,W,b
x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#softmax回归模型
y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float", [None,10])
#计算交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#设置TensorFlow用梯度下降算法以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#初始化变量
init = tf.initialize_all_variables()
#评估模型
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#开启Tesnsorflow
sess = tf.Session()
sess.run(init)
#循环训练模型
for i in range(1000):
  batch = mnist.train.next_batch(100)
  sess.run(train_step,feed_dict={x: batch[0], y_: batch[1]})
#输出结果
print("softmax回归测试MNIST数据集正确率:")
print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))