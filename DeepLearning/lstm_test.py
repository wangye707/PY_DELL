#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : lstm_test.py
# @Author: WangYe
# @Date  : 2018/8/1
# @Software: PyCharm
import numpy
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.activations import relu, tanh
from keras.utils import np_utils
# fix random seed for reproducibility
numpy.random.seed(7)
# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# prepare the dataset of input to output pairs encoded as integers
seq_length = 3
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    print(seq_in, '->', seq_out)
# reshape X to be [samples, time steps, features]
#print(alphabet)
X = numpy.reshape(dataX, (len(dataX), 1, seq_length))
#将X转化为24个元素(abc等换为123数字)，每个元素的长度为1
#在这一个元素中，有存着3个元素，所以为[[[22 23 24]]]的矩阵
#print(dataY)
# normalize
X = X / float(len(alphabet))#X转换为0到1
print(X)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)#转换成一个标签的矩阵
print(y)
# 链接：https://blog.csdn.net/zlrai5895/article/details/79560353
#print(X)
# print(X.shape[0])#23
# print(X.shape[1])#1
# print(X.shape[2])#3
# print(y.shape[0])#23
# print(y.shape[1])#26

# create and fit the model
model = Sequential()
model.add(LSTM(units=32, input_shape=(X.shape[1], X.shape[2])))  # units
model.add(Activation(relu))
model.add(Dense(y.shape[1], activation='softmax'))  #全链接层
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, nb_epoch=500, batch_size=1, verbose=2)
# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))
# demonstrate some model predictions
for pattern in dataX:
    x = numpy.reshape(pattern, (1, 1, len(pattern)))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in, "->", result)
