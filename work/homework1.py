#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : homework1.py
# @Author: WangYe
# @Date  : 2018/3/20
# @Software: PyCharm
# 大数据第一次作业
import os  #操作系统
import struct   #二进制
import random  #导入随机数包
os.chdir('C:\\Users\\wy\\Desktop')
def wf1():
    unc=[]     #存数已经出现的随机数的列表
    fb = open('bin.bin', 'w')#读取二进制文件
    f = open('test1.txt', 'w')#读取文件
    xij=1000000     #循环次数
    #for i in range(1,xij): #循环一万次
    while(xij):
        n = random.randint(1,1000000)  #数字取值范围
        #unc.append(n)
        if (n in unc):
            #xij=xij+1  #如果存在，跳出，循环次数不减
            #xij=xij+1
            continue
        else:
            unc.append(n)
            bin = struct.pack('i', n)  # 转换二进制
            fb.write(str(bin))  # 写入二进制
            f.write(str(n) + ',')  # 写入文件
            xij = xij - 1       #循环次数减一

    f.close()
def wf2():
    f = open('test1.txt', 'r')
    x = []
    for filenum in f.readlines():
        filenum=filenum.replace(","," ") #将，改为空格
        filenum=filenum.split()#分词
    res=[]
    for i in range(len(filenum)):
        m=int(filenum[i]) #将刚读出的lines中每个数字（str型的int）转换为int，
        res.append(m)#写入新的列表
    res1=quick_sort(res,0,99)#快速排序
    #res1=sorted(res)
    f1 = open(('test2.txt'), 'w')
    for i in range(0,len(res1)):
        f1.write(str(res1[i]) + ',')#写入新文件
    f.close()
def quick_sort(lists, left, right):
    # 快速排序
    if left >= right:
        return lists
    key = lists[left]
    low = left
    high = right
    while left < right:
        while left < right and lists[right] >= key:
            right -= 1
        lists[left] = lists[right]
        while left < right and lists[left] <= key:
            left += 1
        lists[right] = lists[left]
    lists[right] = key
    quick_sort(lists, low, left - 1)
    quick_sort(lists, left + 1, high)
    return lists


wf1()
wf2()