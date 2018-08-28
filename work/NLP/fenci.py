#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : fenci.py
# @Author: WangYe
# @Date  : 2018/7/23
# @Software: PyCharm
import jieba#文件名不能也叫jieba.py，否则会报错
import numpy as np
path='C:/Users/wy/Desktop/问题.txt'
path1='C:/Users/wy/Desktop/答案.txt'
def readfile():
    f1=open(path,encoding='utf-8')
    f2=open(path1,encoding='utf-8')
    b=[]#存放问题的数组
    c=[]#存放答案的数组
    for line1 in f1:
        #print(line1)
        b.append(line1)
        #print(len(line1))
        #print(len(b))
    for line2 in f2:
        c.append(line2)
        #print(len(c))
    # print(len(b))
    # print(len(c))
    return b,c    #返回参数问题和答案的列表
def fenci_question():
    # print(readfile()[0])#问题
    question=readfile()[0]
    temp_list=[]
    # print(readfile()[1])#答案
    #for i in range(len(question)):
    for i in range(len(question)):
        #produce1=question[i].replace('，','').replace('、','').replace('？','')
        #print(len(produce1))
       # print(produce1)
        produce2=jieba.cut(question[i])
    #temp_list.append(i)#坐标
        #print(type(temp_list))
        for x in produce2:
            temp_list.append(str(x))
        #temp_list.append(produce2)
        #print('/'.join(produce2))
    print(temp_list)
    path="C:/Users/wy/Desktop/jieba.txt"
    with open(path,'w',encoding='utf-8') as f:
        for i in temp_list:
            f.write(str(i))
            f.write("/")
        # a=np.array(produce2)
        # print(type(a))
    #print(produce2.shape())
    #print(produce2)
    f.close()

if __name__ == '__main__':
    readfile()
    fenci_question()