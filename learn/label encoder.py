#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : label encoder.py
# @Author: WangYe
# @Date  : 2018/4/7
# @Software: PyCharm
# 标记编码器
from sklearn import preprocessing
label_encode=preprocessing.LabelEncoder()#定义标记编码器
input_classes=["a","b","b","c","d","e"]
label_encode.fit(input_classes)#转换数组
print('class mapping')
for i,item in enumerate(label_encode.classes_):#枚举循环遍历
    print(item,'-->',i)
#新数组的转换
labels=['a','c']  #定义一个新的数组
encode_new=label_encode.transform(labels)#转换成label_encode后存到新的encode_new中
print('labels=',labels)
print('encode_new=',list(encode_new))
print('encode_new=',encode_new)#有list和没list的区别
#数字反向转换为单词
encode_word=[2,1,3,1,2,3,0]
encode_num=label_encode.inverse_transform(encode_word)
print(encode_word)
print(encode_num)
