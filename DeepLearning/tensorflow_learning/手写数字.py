#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : 手写数字.py
# @Author: WangYe
# @Date  : 2018/9/14
# @Software: PyCharm
import tensorflow as tf
from tensorflow.examples.tutorials.mnist\
    import input_data
mnist = input_data.read_data_sets\
    ('C:/Users/wy/Desktop/MNIST_data/',one_hot=True)

#每个批次的大小
batch_size = 100
#计算一共多少个批次
n_batch = mnist.train.num_examples  #批次大小

#定义两个placeholder
#                       None:批次的值会传入近来
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

#创建神经网络
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

# 二次代价函数

loss = tf.reduce_mean(tf.square(y-prediction))

#交叉熵代价函数
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits\
#     (labels=y,logits=prediction))
train_step = tf.train.GradientDescentOptimizer(0.2).\
    minimize(loss)

init = tf.global_variables_initializer()

#结果存放在布尔型列表中
                      #比较大小
correct_prediction = tf.equal(tf.argmax(y,1),
         #    argmax求最大值在那个位置(1表示按行查找，0是按列查找)
                              tf.argmax(prediction,1))
#求准确率       平均率
accuracy = tf.reduce_mean(tf.cast #转换类型
                          (correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            # 数据      标签
            batch_xs,batch_ys = \
                mnist.train.next_batch(batch_size)
                        #   每次读取多少照片
            sess.run(train_step,
                     feed_dict={x:batch_xs,y:batch_ys})

            acc = sess.run(accuracy,
                           feed_dict={x:mnist.test.images,
                                      y:mnist.test.labels})
            print('Iter',str(epoch),',Testing Accuracy',
                  str(acc))