#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : keras_test1.py
# @Author: WangYe
# @Date  : 2018/8/1
# @Software: PyCharm
import keras
import numpy as np
np.random.seed(1337)

from keras.datasets import  mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam


# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing，-1 represents the number of samples;1 represents the num of channels,28&28 represents the length,width respectively
X_train = X_train.reshape(-1,1,28,28)  # normalize  （例子的个数,照片的高度）
X_test = X_test.reshape(-1,1,28,28)    # normalize
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


#build neural network

model=Sequential()

model.add(Convolution2D(    #第一层，卷积
    nb_filter=32,   #输出32个滤波器，扫描一个图片，每个滤波器会总结一次图片。所以后面的高度是32层
    nb_col=5,       #
    nb_row=5,       #5*5的filter的宽度
    border_mode='same', #padding方法
    input_shape=(1,28,28) ))#一个高度和28*28的图片

model.add(Activation('relu'))


model.add(MaxPooling2D(
    pool_size=(2,2), #向下取样图片大小
    strides=(2,2),    #跳过长和宽的大小（在取样过程中）
    border_mode='same', #padding method
))

#这是添加第二层神经网络，卷积层，激励函数，池化层
model.add(Convolution2D(64,5,5,border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))

#将经过池化层之后的三维特征，整理成一维。方便后面建立全链接层
model.add(Flatten())
#1024像素
model.add(Dense(1024))  #Dense 全连接层（对上一层的神经元进行全部连接，实现特征的非线性组合）

model.add(Activation('relu'))
#输出压缩到10维，因为有10个标记
model.add(Dense(10))
#使用softmax进行分类
model.add(Activation('softmax'))



# Another way to define your optimize
#优化
adam=Adam(lr=1e-4)

model.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy'])

print('\nTraining-----------')
model.fit(X_train,y_train,nb_epoch=2,batch_size=32)

print('\nTesting------------')
loss,accuracy=model.evaluate(X_test,y_test)


print('test loss: ', loss)
print('test accuracy: ', accuracy)