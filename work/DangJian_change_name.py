#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : DangJian_change_name.py
# @Author: WangYe
# @Date  : 2018/8/29
# @Software: PyCharm
import os
import shutil
path=r'C:\Users\wy\Desktop\电子书文档'
path1=r'C:\Users\wy\Desktop\习近平系列讲话'
def change_docx_name():
    x=-1
    file_name=[]
    for root,dirs,files in os.walk(path):
        if x==-1:
            for i in dirs:
                # print(i)
                file_name.append(i)
        else:
            #print(files)
            dir_name=file_name[x]
            for i in range(len(files)):
                doc_path=os.path.join(root,files[i])
                print(doc_path)
                #print(type(dir_name))
                new_name=dir_name+'  '+files[i]
                new_path=os.path.join(root,new_name)
                os.renames(doc_path, new_path)
                #shutil.copyfile(new_path,path1)
        x=x+1
def change_file_name():
    x = -1
    file_name = []
    for root, dirs, files in os.walk(path):
        if x == -1:
            for i in dirs:
                # print(i)
                file_name.append(i)
        else:
            # print(files)
            dir_name = file_name[x]
            for i in range(len(files)):
                #print(root+files[i])
                old_path=os.path.join(root,files[i])
                #print(old_path)
                shutil.move(old_path,path1)
        x = x + 1

#change_docx_name()
change_file_name()




