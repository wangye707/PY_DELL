#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : CSV_Cleaning.py
# @Author: WangYe
# @Date  : 2018/7/20
# @Software: PyCharm
import csv
import numpy as np
# import sys
path = 'C:/Users/wy/Desktop/DJ.materials.csv'
def readCSV(path):#读取答案
    csvfile = open(path, encoding='UTF-8')  # 打开一个文件
    reader = csv.DictReader(csvfile)  # 返回的可迭代类型
    column = [row['content'] for row in reader]
    print(len(column))
    return column

def readCSV1(path):#读取问题
    csvfile = open(path, encoding='UTF-8')  # 打开一个文件
    reader = csv.DictReader(csvfile)  # 返回的可迭代类型
    column1 = [row['name'] for row in reader]
    print(len(column1))
    return column1

def writeTXT(): #写入答案
    path2='C:/Users/wy/Desktop/答案.txt'
    with open(path2, 'w', newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        # #print(readCSV(path))
        #shuju=readCSV(path)#读出数组
        shuju=readCSV(path)
        #content=np.array(shuju).reshape(len(shuju),1)
        content=np.array(shuju).reshape(len(shuju),1)
        # for i in range(len(content)):
        #      writer.writerows(content[i])
        for i in range(len(content)):
             writer.writerow(content[i])

def writeTXT1():
    path2='C:/Users/wy/Desktop/问题.txt'
    with open(path2, 'w', newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        # #print(readCSV(path))
        #shuju=readCSV(path)#读出数组
        shuju1=readCSV1(path)
        #content=np.array(shuju).reshape(len(shuju),1)
        name=np.array(shuju1).reshape(len(shuju1),1)
        # for i in range(len(content)):
        #      writer.writerows(content[i])
        for i in range(len(name)):
             writer.writerow(name[i])
if __name__ == '__main__':
    writeTXT1()
    writeTXT()