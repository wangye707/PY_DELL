#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : 遍历文件夹的子文件.py
# @Author: WangYe
# @Date  : 2018/4/10
# @Software: PyCharm
import os
strpath='C:/Users/wy/Desktop/test'
os.chdir(strpath)
a=[]
for (root, dirs, files) in os.walk(strpath):  #列出windows目录下的所有文件和文件名
    for filename in files:
        print(os.path.join(root,filename))
        pathload=str(os.path.join(root,filename))
        #print(pathload)
        f=open(pathload,'r')   #不能用os.open
        for filenum in f.readlines():
            filenum = filenum.replace(",", " ")  # 将，改为空格
            filenum = filenum.split()  # 分词
            # i=0
            for i,words in enumerate(filenum):
                a.append(words)

            print(words)
import  os
folder='C:/Users/wy/Desktop/test'
fs=os.listdir(folder)
allTxt=''
for i in fs:
    name=folder+'/'+i
    f=os.listdir(name)
    for  j in f:
        _name=name+'/'+j

        with open(_name,'r') as f1:
            allTxt+=f1.read()+'\n'
        break
print(allTxt)
