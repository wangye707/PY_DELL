#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : shibada_new_word.py
# @Author: WangYe
# @Date  : 2018/8/31
# @Software: PyCharm
import os
import re
path=r'C:\Users\wy\Desktop\Data_Test\1.txt'
path1=r'C:\Users\wy\Desktop\Data_Test\2.txt'
compiler = re.compile(r'\d+\.')
content2=[]
with open(path,'r') as f:
    content=f.read()
    #print(type(content))
    content1=compiler.split(content)
    # print(content1)
    # content2.append(content1)
    #print(type(content1))
    #temp = eval(content[content.index('【'):])
    #print(content1)
    #str(content1)
    #f.write(str(content1))
with open(path1,'w') as f:
    f.write(str(content1))