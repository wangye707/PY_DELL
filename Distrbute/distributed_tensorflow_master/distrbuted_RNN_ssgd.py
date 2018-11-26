#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : distrbuted_RNN.py
# @Author: WangYe
# @Date  : 2018/11/5
# @Software: PyCharm

import tensorflow as tf
from sklearn.model_selection import train_test_split
import collections
import nltk#用来分词
import numpy as np#用来统计词频
import os
import time
# Define parameters
FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_float('learning_rate', 0.00003, 'Initial learning rate.')
# tf.app.flags.DEFINE_integer('steps_to_validate', 1000,
#                      'Steps to validate and print loss')
#
# For distributed
tf.app.flags.DEFINE_string("ps_hosts","192.168.1.124:11111",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "192.168.1.124:11112",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("issync", 0, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")
tf.app.flags.DEFINE_string("cuda", "", "specify gpu")
#FLAGS = tf.app.flags.FLAGS
if FLAGS.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda
# = FLAGS.learning_rate
#steps_to_validate = FLAGS.steps_to_validate
'''文本预处理'''
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
'''文本预处理完毕'''

learning_rate = 0.001
batch_size = 50
train_step = 10000
display_step = 100

frame_size = 41
# frame_size: 序列里面每一个分量的单词数量
sequence_length = 7086
# sequence_size: 每个样本序列的长度
hidden_num = 128
n_classes = 2
start=time.time()
def RNN(x,weights,bias):
    x=tf.reshape(x,shape=[-1,41,1])
    # print(type(x))
    rnn_cell=tf.nn.rnn_cell.BasicRNNCell(hidden_num)
    init_state=tf.zeros(shape=[batch_size,rnn_cell.state_size])
    # 其实这是一个深度RNN网络,对于每一个长度为n的序列[x1,x2,x3,...,xn]的每一个xi,都会在深度方向跑一遍RNN,跑上hidden_num个隐层单元
    output,states=tf.nn.dynamic_rnn(rnn_cell,x,dtype=tf.float32)
    #print(output[1].shpae)
    return tf.nn.softmax(tf.matmul(output[:,-1],weights)+bias,1)

def loss1(label, pred):
    return tf.square(label - pred)

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    issync = FLAGS.issync
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            global_step = tf.Variable(0, name='global_step', trainable=False)

            x = tf.placeholder(dtype=tf.float32, shape=[None, 41], name="inputx")
            y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="expected_y")

            weights = tf.Variable(tf.truncated_normal(shape=[hidden_num, n_classes], stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=[n_classes]))
            predy = RNN(x, weights, bias)

            #loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predy, labels=y))
            #train = tf.train.AdamOptimizer(train_rate).minimize(cost)
            loss_value = loss1(y, predy)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            correct_pred = tf.equal(tf.argmax(predy, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.to_float(correct_pred))

            grads_and_vars = optimizer.compute_gradients(loss_value)
            if issync == 1:
                # 同步模式计算更新梯度
                rep_op = tf.train.SyncReplicasOptimizer(optimizer,
                                                        replicas_to_aggregate=len(
                                                            worker_hosts),
                                   #                     replica_id=FLAGS.task_index,
                                                        total_num_replicas=len(
                                                            worker_hosts),
                                                        use_locking=True)
                train_op = rep_op.apply_gradients(grads_and_vars,
                                                  global_step=global_step)
                init_token_op = rep_op.get_init_tokens_op()
                chief_queue_runner = rep_op.get_chief_queue_runner()
            else:
                # 异步模式计算更新梯度
                train_op = optimizer.apply_gradients(grads_and_vars,
                                                     global_step=global_step)

            init_op = tf.initialize_all_variables()

            #saver = tf.train.Saver()
            tf.summary.scalar('cost', loss_value)
            summary_op = tf.summary.merge_all()

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                               #  logdir="./checkpoint/",
                                 init_op=init_op,
                                 summary_op=None,
                               #  saver=saver,
                                 global_step=global_step,
                                # save_model_secs=60
                                 )

        with sv.prepare_or_wait_for_session(server.target) as sess:
            # 如果是同步模式
            if FLAGS.task_index == 0 and issync == 1:
                sv.start_queue_runners(sess, [chief_queue_runner])
                sess.run(init_token_op)
            step = 0
            testx, testy = Xtest, ytest
            while step < train_step:
                batch_x, batch_y = Xtrain, ytrain
                _, loss_v, step = sess.run([train_op, loss_value, global_step],
                                           feed_dict={x: batch_x, y: batch_y})
                if step % display_step == 0:
                    acc, loss = sess.run([accuracy, loss_value], feed_dict={x: testx, y: testy})
                    print('train:', step, acc, loss)

                if step % display_step == 0:
                    acc, loss = sess.run([accuracy, loss_value], feed_dict={x: testx, y: testy})
                    print('test:', step, acc, loss)

        #sv.stop()
        end=time.time()
        print("costing time:", end - start)



if __name__ == "__main__":
    tf.app.run()