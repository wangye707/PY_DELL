#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : RNN_test.py
# @Author: WangYe
# @Date  : 2018/8/2
# @Software: PyCharm
import numpy as np
np.random.seed(1337)

from keras.datasets import  mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten,SimpleRNN
from keras.optimizers import Adam

TIME_STEPS=28#时间长度  28行图片
INPUT_SIZE=28#
BATCH_SIZE=50#每一批训练图片数量
BATCH_INDEX=0 #用它最后生成数据
OUTPUT_SIZE=10#用来存放最后预测数字
CELL_SIZE=50  #
LR=0.001   #学习速率
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing，-1 represents the number of samples;1 represents the num of channels,28&28 represents the length,width respectively
X_train = X_train.reshape(-1,1,28,28)/255  # normalize  （例子的个数,照片的高度）,除以255是让它在0到1之间
X_test = X_test.reshape(-1,1,28,28)/255 # normalize
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


#创建RNN模型
model=Sequential()
#开始定义RNN
model.add(SimpleRNN(
    batch_input_shape=(BATCH_SIZE,TIME_STEPS,INPUT_SIZE),
    output_dim=CELL_SIZE,

))
'''输出层'''
model.add(Dense(OUTPUT_SIZE))   #输出10个单位
model.add(Activation('softmax'))   #默认谈好：-1到1之间

#优化
adam=Adam(LR)

model.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy'])
#测试
for step in range(4001):
    #data shape=(batch_num,steps,inputs/outputs)
    X_batch=X_train[BATCH_INDEX:BATCH_SIZE+BATCH_INDEX,:,:]   #一批一批的截取
    Y_batch=y_train[BATCH_INDEX:BATCH_SIZE+BATCH_INDEX,:]
    cost=model.train_on_batch(X_batch,Y_batch)
    BATCH_INDEX+=BATCH_SIZE
    BATCH_INDEX=0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX
    if step %500 == 0:
        cost,accuracy=model.evaluate(X_test,y_test,batch_size=y_test.shape[0],verbose=False)
        print('test cost:',cost,'test accuracy:',accuracy)