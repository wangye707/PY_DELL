# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 09:50:43 2017

@author: Peter
"""
import tensorflow as tf

class QaNet(object):
	def __init__(self, batchSize, timeStep, embeddings, embeddingSize, rnnSize, margin, attention_matrix_size):
		self.batchSize = batchSize
		self.timeStep = timeStep
		self.embeddings = embeddings
		self.embeddingSize= embeddingSize
		self.rnnSize = rnnSize
		self.margin = margin
		self.attention_matrix_size = attention_matrix_size

		self.inputQuestions = tf.placeholder(tf.int32,shape=[None,self.timeStep])
		self.inputTrueAnswers = tf.placeholder(tf.int32,shape=[None,self.timeStep])
		self.inputFalseAnswers = tf.placeholder(tf.int32,shape=[None,self.timeStep])
		self.inputTestQuestions  = tf.placeholder(tf.int32,shape=[None,self.timeStep])
		self.inputTestAnswers  = tf.placeholder(tf.int32,shape=[None,self.timeStep])
		self.lr = tf.placeholder(tf.float32)

		with tf.name_scope("embedding_layer"):
			# 将词索引映射到词向量
			tfEmbedding = tf.Variable(tf.to_float(self.embeddings),trainable=True,name="W")
			questions = tf.nn.embedding_lookup(tfEmbedding,self.inputQuestions)
			trueAnswers = tf.nn.embedding_lookup(tfEmbedding,self.inputTrueAnswers)
			falseAnswers = tf.nn.embedding_lookup(tfEmbedding,self.inputFalseAnswers)
			testQuestions = tf.nn.embedding_lookup(tfEmbedding,self.inputTestQuestions)
			testAnswers = tf.nn.embedding_lookup(tfEmbedding,self.inputTestAnswers)

		with tf.variable_scope("lstm_layer", reuse=None):
			question = self.biLSTMCell(questions, self.rnnSize)
		with tf.variable_scope("lstm_layer", reuse=True):
			trueAnswer = self.biLSTMCell(trueAnswers, self.rnnSize)
			falseAnswer = self.biLSTMCell(falseAnswers, self.rnnSize)
			testQuestion = self.biLSTMCell(testQuestions, self.rnnSize)
			testAnswer = self.biLSTMCell(testAnswers, self.rnnSize)

		with tf.name_scope("att_weight"):
			# attention params
			att_W = {
				'Wam': tf.Variable(tf.truncated_normal([2 * self.rnnSize, self.attention_matrix_size], stddev=0.1)),
				'Wqm': tf.Variable(tf.truncated_normal([2 * self.rnnSize, self.attention_matrix_size], stddev=0.1)),
				'Wms': tf.Variable(tf.truncated_normal([self.attention_matrix_size, 1], stddev=0.1))
			}
		true_feat_q, true_feat_a = self.get_feature(question,trueAnswer,att_W)
		false_feat_q, false_feat_a = self.get_feature(question,falseAnswer,att_W)
		self.true_sim = self.feature2cos_sim(true_feat_q, true_feat_a)
		self.false_sim = self.feature2cos_sim(false_feat_q, false_feat_a)
		self.loss, self.acc = self.cal_loss_and_acc(self.true_sim, self.false_sim)
		self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

		test_feat_q, test_feat_a = self.get_feature(testQuestion,testAnswer,att_W)
		self.scores = self.feature2cos_sim(test_feat_q, test_feat_a)
		
	def biLSTMCell(self, x, hiddenSize):
		input_x = tf.transpose(x, [1, 0, 2])
		input_x = tf.unstack(input_x)
		lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hiddenSize, forget_bias=1.0, state_is_tuple=True)
		lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hiddenSize, forget_bias=1.0, state_is_tuple=True)
		output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input_x, dtype=tf.float32)
		output = tf.stack(output)
		output = tf.transpose(output, [1, 0, 2])
		return output

	def max_pooling(self, lstm_out):
		height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])       # (step, length of input for one step)

		# do max-pooling to change the (sequence_length) tensor to 1-length tensor
		lstm_out = tf.expand_dims(lstm_out, -1)
		output = tf.nn.max_pool(
			lstm_out,
			ksize=[1, height, 1, 1],
			strides=[1, 1, 1, 1],
			padding='VALID')

		output = tf.reshape(output, [-1, width])

		return output

	def get_feature(self, input_q, input_a, att_W):
		h_q = int(input_q.get_shape()[1]) # length of question
		w = int(input_q.get_shape()[2]) # length of input for one step
		h_a = int(input_a.get_shape()[1]) # length of answer

		output_q = self.max_pooling(input_q) # (b,w)

		reshape_q = tf.expand_dims(output_q, 1) # (b,1,w)  b:batch size
		reshape_q = tf.tile(reshape_q, [1, h_a, 1]) # (b,h_a,w)
		reshape_q = tf.reshape(reshape_q, [-1, w]) # (b*h_a, w)
		reshape_a = tf.reshape(input_a, [-1, w]) # (b*h_a,w)

		M = tf.tanh(tf.add(tf.matmul(reshape_q, att_W['Wqm']), tf.matmul(reshape_a, att_W['Wam'])))
		M = tf.matmul(M, att_W['Wms']) # (b*h_a,1)

		S = tf.reshape(M, [-1, h_a]) # (b,h_a)
		S = tf.nn.softmax(S) # (b,h_a)

		S_diag = tf.matrix_diag(S) # (b,h_a,h_a)
		attention_a = tf.matmul(S_diag, input_a) # (b,h_a,w)

		output_a = self.max_pooling(attention_a) # (b,w)

		return tf.tanh(output_q), tf.tanh(output_a)

	def feature2cos_sim(self, feat_q, feat_a):
		norm_q = tf.sqrt(tf.reduce_sum(tf.multiply(feat_q, feat_q), 1))
		norm_a = tf.sqrt(tf.reduce_sum(tf.multiply(feat_a, feat_a), 1))
		mul_q_a = tf.reduce_sum(tf.multiply(feat_q, feat_a), 1)
		cos_sim_q_a = tf.div(mul_q_a, tf.multiply(norm_q, norm_a))
		return cos_sim_q_a

	def cal_loss_and_acc(self, trueCosSim, falseCosSim):
		# the target function 
		zero = tf.fill(tf.shape(trueCosSim), 0.0)
		margin = tf.fill(tf.shape(trueCosSim), self.margin)
		with tf.name_scope("loss"):
		    losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(trueCosSim, falseCosSim)))
		    loss = tf.reduce_sum(losses) 
		# cal accurancy
		with tf.name_scope("acc"):
		    correct = tf.equal(zero, losses)
		    acc = tf.reduce_mean(tf.cast(correct, "float"), name="acc")
		return loss, acc