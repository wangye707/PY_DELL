#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : keras_test2.py
# @Author: WangYe
# @Date  : 2019/1/8
# @Software: PyCharm
import tensorflow as tf
import numpy as np
np.random.seed(1337)  # for reproducibility
import  tensorflow as tf
from  tensorflow.examples.tutorials.mnist import  input_data
from tensorflow.contrib import  rnn

mnist=input_data.read_data_sets(r"./wy_data",one_hot=True)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# 全局变量
batch_size = 128
nb_classes = 10
epochs = 12
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 根据不同的backend定下不同的格式
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 转换为one_hot类型
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#构建模型
model = Sequential()
"""
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='same',
                        input_shape=input_shape))
"""
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='same',
                        input_shape=input_shape)) # 卷积层1
model.add(Activation('relu')) #激活层
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]))) #卷积层2
model.add(Activation('relu')) #激活层
model.add(MaxPooling2D(pool_size=pool_size)) #池化层
model.add(Dropout(0.25)) #神经元随机失活
model.add(Flatten()) #拉成一维数据
model.add(Dense(128)) #全连接层1
model.add(Activation('relu')) #激活层
model.add(Dropout(0.5)) #随机失活
model.add(Dense(nb_classes)) #全连接层2
model.add(Activation('softmax')) #Softmax评分

#编译模型

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
#训练模型
# model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
#           verbose=1, validation_data=(X_test, Y_test))







x=X_train
y=Y_train


model.fit\
    (
        x=None,y=None,
     batch_size=batch_size,epochs=epochs,
          verbose=1, validation_data=(X_test, Y_test),steps_per_epoch=1)
#
# #评估模型
# step = 0
# while step<500:
#     batch_x,batch_y=mnist.train.next_batch(batch_size)
# #    batch_x=tf.reshape(batch_x,shape=[batch_size,sequence_length,frame_size])
#     _loss,__=sess.run([cost,train],feed_dict={x:batch_x,y:batch_y})
#     if step % display_step ==0:
#
#         acc,loss=sess.run([accuracy,cost],feed_dict={x:testx,y:testy})
#         print(step,acc,loss)


score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])