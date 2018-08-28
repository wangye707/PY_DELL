#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : keras_test.py
# @Author: WangYe
# @Date  : 2018/8/1
# @Software: PyCharm
from keras.models import Sequential
'''
Keras有两种类型的模型，序贯模型（Sequential）和函数式模型（Model），函数式模型应用更为广泛，
       序贯模型是函数式模型的一种特殊情况。
a）序贯模型（Sequential):单输入单输出，一条路通到底，层与层之间只有相邻关系，没有跨层连接。
       这种模型编译速度快，操作也比较简单
b）函数式模型（Model）：多输入多输出，层与层之间任意连接。这种模型编译速度慢。
'''
from keras.layers.core import Dense, Dropout, Activation
#指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。
#dropout是CNN中防止过拟合提高效果的一个大杀器
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy
'''
    第一步：选择模型
'''
model = Sequential()
'''
   第二步：构建网络层
'''
model.add(Dense(500,input_shape=(784,))) # 输入层，28*28=784
model.add(Activation('tanh')) # 激活函数是tanh
#每个神经元都有激活函数： linear，sigmoid，tanh，softmax，LeakyReLU和PReLU
model.add(Dropout(0.5)) # 采用50%的dropout,防止过拟合

model.add(Dense(500)) # 隐藏层节点500个
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(10)) # 输出结果是10个类别，所以维度是10
model.add(Activation('softmax')) # 最后一层用softmax作为激活函数

'''
   第三步：编译
'''
'''
参数：
optimizer：指定模型训练的优化器；
loss：目标函数；
class_mode: ”categorical”和”binary”中的一个，只是用来计算分类的精确度或using the predict_classes method
theano_mode: Atheano.compile.mode.Mode instance controllingspecifying compilation options 
'''
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # 优化函数，设定学习率（lr）等参数
model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode='categorical') # 使用交叉熵作为loss函数

'''
   第四步：训练
   .fit的一些参数
   batch_size：对总的样本数进行分组，每组包含的样本数量
   epochs ：训练次数
   shuffle：是否把数据随机打乱之后再进行训练
   validation_split：拿出百分之多少用来做交叉验证
   verbose：屏显模式 0：不输出  1：输出进度  2：输出每次的训练结果
'''
(X_train, y_train), (X_test, y_test) = mnist.load_data() # 使用Keras自带的mnist工具读取数据（第一次需要联网）
# 由于mist的输入数据维度是(num, 28, 28)，这里需要把后面的维度直接拼起来变成784维
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
Y_train = (numpy.arange(10) == y_train[:, None]).astype(int)
Y_test = (numpy.arange(10) == y_test[:, None]).astype(int)

model.fit(X_train,Y_train,batch_size=200,epochs=50,shuffle=True,verbose=0,validation_split=0.3)
'''
X：训练数据
y : 标签
batch_size : 每次训练和梯度更新块的大小。
nb_epoch: 迭代次数。
verbose : 进度表示方式。0表示不显示数据，1表示显示进度条，2表示用只显示一个数据。
callbacks : 回调函数列表。就是函数执行完后自动调用的函数列表。
validation_split : 验证数据的使用比例。
validation_data : 被用来作为验证数据的(X, y)元组。会代替validation_split所划分的验证数据。
shuffle : 类型为boolean或 str(‘batch’)。是否对每一次迭代的样本进行shuffle操作（可以参见博文Theano学习笔记01--Dimshuffle()函数）。’batch’是一个用于处理HDF5（keras用于存储权值的数据格式）数据的特殊选项。
show_accuracy:每次迭代是否显示分类准确度。
class_weigh : 分类权值键值对。
sample_weight : list or numpy array with1:1 mapping to the training samples,
    used for scaling the loss function (duringtraining only). For time-distributed data,
    there is one weight per sample pertimestep, i.e. if your output data is shaped(nb_samples, timesteps, output_dim), 
    your mask should be of shape (nb_samples, timesteps, 1). This allows you to maskout or reweight individual output timesteps, 
    which is useful in sequence tosequence learning.
'''
model.evaluate(X_test, Y_test, batch_size=200, verbose=0)

'''
    第五步：输出
'''
print("test set")
scores = model.evaluate(X_test,Y_test,batch_size=200,verbose=0)
print("")
print("The test loss is %f" % scores)
result = model.predict(X_test,batch_size=200,verbose=0)

result_max = numpy.argmax(result, axis = 1)
test_max = numpy.argmax(Y_test, axis = 1)

result_bool = numpy.equal(result_max, test_max)
true_num = numpy.sum(result_bool)
print("")
print("The accuracy of the model is %f" % (true_num/len(result_bool)))