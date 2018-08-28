#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : os_walk_test.py
# @Author: WangYe
# @Date  : 2018/8/28
# @Software: PyCharm
# coding: utf-8
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from time import time
import os
import sys
import fnmatch
import win32com.client
PATH = os.path.abspath(os.path.dirname(sys.argv[0]))
print(PATH)
# PATH_DATA = os.path.abspath\
#                 (os.path.dirname(sys.argv[0])) + "\data"
#print(PATH_DATA)
PATH_DATA=r'C:\Users\wy\Desktop\data\8原文文档'
# 主要执行函数
'''将docx转化为txt'''
def readfile():
    docx_name=[]   #存储文件名
    filename=[]   #存储文件种类名
    all_file=[]    #存储所有文件
    #temp1=[]
    content=[]
    print('文件开始读取')
    x=-1
    try:
        for root, dirs, files in os.walk(PATH_DATA):
            # if x==-1:
            #     #print(dirs)
            #     for name in dirs:#访问第一层文件夹
            #         #print(name)
            #         filename.append(name)
            #         temp_content=[]
            # else:

                #print(files)
            # temp=[]
            #print('已经读取到', filename[x], '文件夹')
            temp_content = []
            for every_docx_name in files:
                # print('hang',hang)
                every_file_path = \
                    os.path.join(root, every_docx_name)
                # print(every_file_path)
                # print(every_file_path)
                # print(filename)
                with open(every_file_path,encoding='utf-8') as f:
                    temp = f.read()
                    print(
                        #'name', filename[x],
                          'docname', every_docx_name,
                          'content', len(temp))
                temp_content.append(temp)
            # print(len(temp_content))
            content.append(temp_content)
            # print(every_file_path)


            # x=x+1



            #print(files)


    finally:
        print("文件读取完成")
        #return filename,all_file,docx_name
'''文件操作结束，开始es'''
readfile()