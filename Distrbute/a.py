#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : a.py
# @Author: WangYe
# @Date  : 2018/10/24
# @Software: PyCharm
# coding=utf-8
# 上面是因为worker计算内容各不相同，不过再深度学习中，一般每个worker的计算内容是一样的，
# 以为都是计算神经网络的每个batch 前向传导，所以一般代码是重用的
import tensorflow as tf

# 现在假设我们有A、B台机器，首先需要在各台机器上写一份代码，并跑起来，各机器上的代码内容大部分相同
# ，除了开始定义的时候，需要各自指定该台机器的task之外。以机器A为例子，A机器上的代码如下：
cluster = tf.train.ClusterSpec({
    "worker": [
        "192.168.11.105:1234",  # 格式 IP地址：端口号，第一台机器A的IP地址 ,在代码中需要用这台机器计算的时候，就要定义：/job:worker/task:0
    ],
    "ps": [
        "192.168.11.130:2223"  # 第四台机器的IP地址 对应到代码块：/job:ps/task:0
    ]})

# 不同的机器，下面这一行代码各不相同，server可以根据job_name、task_index两个参数，查找到集群cluster中对应的机器

isps = False
if isps:
    server = tf.train.Server(cluster, job_name='ps', task_index=0)  # 找到‘worker’名字下的，task0，也就是机器A
    server.join()
else:
    server = tf.train.Server(cluster, job_name='worker', task_index=0)  # 找到‘worker’名字下的，task0，也就是机器A
    with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:0', cluster=cluster)):
        w = tf.get_variable('w', (2, 2), tf.float32, initializer=tf.constant_initializer(2))
        b = tf.get_variable('b', (2, 2), tf.float32, initializer=tf.constant_initializer(5))
        addwb = w + b
        mutwb = w * b
        divwb = w / b

saver = tf.train.Saver()
summary_op = tf.merge_all_summaries()
init_op = tf.initialize_all_variables()
sv = tf.train.Supervisor(init_op=init_op, summary_op=summary_op, saver=saver)
with sv.managed_session(server.target) as sess:
    while 1:
        print(sess.run([addwb, mutwb, divwb]))
