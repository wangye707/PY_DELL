# 存放数据的路径
DATA_PATH = "./data/"
checkpoint_path="./model"
hidden_size = 200    # 隐藏层，用于记忆和储存过去状态的节点个数
num_layers = 2  # LSTM结构的层数为2层，前一层的LSTM的输出作为后一层的输入
vocab_size = 10000  # 词典大小，可以存储10000个

learning_rate = 1.0  # 初始学习率
train_batch_size = 20  # 训练batch大小
train_num_step = 35  # 一个训练序列长度
num_epoch = 2
max_grad_norm = 5  # 用于控制梯度膨胀（误差对输入层的偏导趋于无穷大）

import os
os.system("rm -rf model && mkdir model")

import collections
import tensorflow as tf
import numpy as np
import sys

Py3 = sys.version_info[0] == 3

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()

def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  train_path = os.path.join(data_path, "ptb.train.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, vocabulary

def ptb_producer(raw_data, batch_size, num_steps, train_steps,all_epoch):
    data_len =len(raw_data)
    epoch=(data_len-1)//(batch_size*num_steps)
    if data_len>=(train_steps+1)*batch_size*num_steps+1 or epoch>=all_epoch:
        train_steps%=all_epoch
        x=[]
        y=[]
        for i in range(train_steps*batch_size*num_steps,(train_steps+1)*batch_size*num_steps):
            x.append(raw_data[i])
            y.append(raw_data[i+1])
        x=np.reshape(x,[batch_size, num_steps])
        y=np.reshape(y,[batch_size, num_steps])
        return x,y
    return 0


def build_model():
    #global initial_state
    g=tf.Graph()
    with g.as_default():
        # 定义输入层，输入层维度为batch_size * num_steps
        input_data = tf.placeholder(tf.int32, [train_batch_size, train_num_step],name="input")
        # 定义正确输出
        targets = tf.placeholder(tf.int32, [train_batch_size, train_num_step],name="label")
        # 定义lstm结构
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        # 使用dropout
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)    # 实现多层LSTM

        # 将lstm中的状态初始化为全0数组，BasicLSTMCell提供了zero_state来生成全0数组
        # batch_size给出了一个batch的大小
        initial_state = cell.zero_state(train_batch_size, tf.float32)
        # 生成单词向量，单词总数为10000，单词向量维度为hidden_size200，所以词嵌入总数embedding为
        embedding = tf.Variable(tf.random_uniform([vocab_size, hidden_size], -1.0, 1.0))
        # lstm输入单词为batch_size*num_steps个单词，则输入维度为batch_size*num_steps*hidden_size
        # embedding_lookup为将input_data作为索引来搜索embedding中内容，若input_data为[0,0],则输出为embedding中第0个词向量
        inputs = tf.nn.embedding_lookup(embedding, input_data)
        # 在训练时用dropout
        inputs = tf.nn.dropout(inputs, keep_prob)

        # 输出层
        outputs = []
        # state为不同batch中的LSTM状态，初始状态为0
        state = initial_state
        for time_step in range(train_num_step):
            if time_step > 0:
                # variables复用
                tf.get_variable_scope().reuse_variables()
            # 将当前输入进lstm中,inputs输入维度为batch_size*num_steps*hidden_size
            cell_output, state = cell(inputs[:, time_step, :], state)
            # 输出队列
            outputs.append(cell_output)

        # 输出队列为[batch, hidden_size*num_steps]，在改成[batch*num_steps, hidden_size]
        # [-1, hidden_size]中-1表示任意数量的样本
        output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])

        # lstm的输出经过全连接层得到最后结果，最后结果的维度是10000，softmax后表明下一个单词的位置（概率大小）
        weight = tf.Variable(tf.truncated_normal(shape=[hidden_size, vocab_size], stddev = 0.1),name="weight")
        bias = tf.Variable(tf.zeros(shape=[vocab_size]),name="bias")
        logits = tf.matmul(output, weight) + bias  # 预测的结果

        # 交叉熵损失，tensorflow中有sequence_loss_by_example来计算一个序列的交叉熵损失和
        # tf.reshape将正确结果转换为一维的,tf.ones建立损失权重，所有权重都为1，不同时刻不同batch的权重是一样的
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(targets, [-1])],
                                                      [tf.ones([train_batch_size * train_num_step], dtype=tf.float32)])

        # 每个batch的平均损失,reduce_sum计算loss总和
        cost = tf.div(tf.reduce_sum(loss),train_batch_size,name="loss")
        final_state = state

        trainable_variables = tf.trainable_variables()
        # 使用clip_by_global_norm控制梯度大小，避免梯度膨胀
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, trainable_variables), max_grad_norm)
        # 梯度下降优化
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # 训练步骤,apply_gradients将计算出的梯度应用到变量上
        # zip将grads和trainable_variables中每一个打包成元组
        # a = [1,2,3]， b = [4,5,6]， zip(a, b)： [(1, 4), (2, 5), (3, 6)]
        global_step = tf.Variable(0, trainable=False, name="global_step")
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables),global_step=global_step,name="train_op")
    return g

def train_model(graph):
    train_data,_ = ptb_raw_data(DATA_PATH)
    all_epoch=(len(train_data)-1)//(train_batch_size*train_num_step)

    x = graph.get_tensor_by_name("input:0")
    y_ = graph.get_tensor_by_name("label:0")
    global_step = graph.get_tensor_by_name("global_step:0")
    train_op = graph.get_tensor_by_name("train_op:0")
    cross_entropy = graph.get_tensor_by_name("loss:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")

    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    with tf.Session(graph=graph,config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for step in range(15):
            x1, y1 = ptb_producer(train_data, train_batch_size, train_num_step,step,all_epoch)
            feed_dict = {x: x1, y_: y1,keep_prob:0.5}
            sess.run(train_op, feed_dict = feed_dict)
            train_loss,g_strp= sess.run([cross_entropy,global_step],feed_dict=feed_dict)
            current_info = "step %d, train_loss %.4f global_step %d" % (step, train_loss,g_strp)
            print(current_info)
            step+=1
        saver.save(sess, checkpoint_path + "/model.ckpt", global_step=global_step)

def main():
    graph=build_model()
    train_model(graph)

if __name__ == "__main__":
    main()