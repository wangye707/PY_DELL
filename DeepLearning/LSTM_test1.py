#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : LSTM_test1.py
# @Author: WangYe
# @Date  : 2018/8/2
# @Software: PyCharm
import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM,TimeDistributed,Dense
from keras.optimizers import Adam

BATCH_START=0
TIME_STEPS=20#一步20，等于切分，每一小块20长度
INPUT_SIZE=1#一个输入点对应一个数据
OUTPUT_SIZE=1#一个输出点对应一个数据
BATCH_SIZE=50#每一批训练图片数量
CELL_SIZE=20 #
LR=0.006  #学习速率