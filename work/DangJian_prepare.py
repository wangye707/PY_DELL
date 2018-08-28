#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : DangJian_prepare.py
# @Author: WangYe
# @Date  : 2018/8/27
# @Software: PyCharm
# coding: utf-8
import numpy
import jieba
import os
from threading import Thread
import time
from os import walk
#import CSVOP
import codecs
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from time import time
import os
import sys
import fnmatch
import win32com.client
PATH = os.path.abspath(os.path.dirname(sys.argv[0]))
#print(PATH)
PATH_DATA=r'C:\Users\wy\Desktop\data\DangJian_prepare'
# 主要执行函数
'''将docx转化为txt'''
def docx_to_text():
    wordapp = win32com.client.\
        gencache.EnsureDispatch("Word.Application")
    try:
        for root, dirs, files\
                in os.walk(PATH_DATA):
            #root代表路径,dirs代表目录，files代表文件名
            print(root,dirs,files)
            for _dir in dirs:   #若是目录，跳过
                pass
            for _file in files:  #若是文件，转化为txt
                if not fnmatch.\
                        fnmatch(_file, '*.docx'):
                    #若是docx结尾，才进行操作
                    continue
                word_file = os.path.join(root, _file)
                wordapp.Documents.Open(word_file)
                #打开word文件
                docastxt = word_file[:-4] + 'txt'
                #新建txt的文件名
                wordapp.ActiveDocument\
                    .SaveAs(
                    docastxt,
                    FileFormat=
                    win32com.client.constants.wdFormatText)
                wordapp.ActiveDocument.Close()
    finally:
        wordapp.Quit()
    print("well done!")
'''遍历TXT文件并且调用es插入数据'''
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
            #print(x)
            if x==-1:
                #print(dirs)
                for name in dirs:#访问第一层文件夹
                    #print(name)
                    filename.append(name)
                    temp_content=[]
            else:

                #print(files)
                #temp=[]
                print('已经读取到', filename[x], '文件夹')
                cishu=1
                temp_content = []
                for every_docx_name in files:
                    # print('hang',hang)
                    every_file_path = \
                        os.path.join(root, every_docx_name)
                    # print(every_file_path)
                    # print(every_file_path)
                    # print(filename)
                    try:#判断编码问题
                        with codecs.open(every_file_path,
                                         encoding='gbk') as f:
                            temp = f.read()
                            #print('++++',temp)
                            '''输出各个文档的信息'''
                            # print('name', filename[x],
                            #       'docname', every_docx_name,
                            #       'content', len(temp))
                            '''执行文档插入操作'''
                            obj.bulk_Index_Data \
                            (filename[x],
                             every_docx_name,
                             temp,
                             cishu)
                        cishu=cishu+1

                    except UnicodeDecodeError:
                        #如果不是gbk，执行utf-8操作
                        with codecs.open(every_file_path,
                                         encoding='utf-8') as f:
                            temp = f.read()
                            # print('++++',temp)
                            # print('name', filename[x],
                            #       'docname', every_docx_name,
                            #       'content', len(temp))
                            # ''''''
                            obj.bulk_Index_Data \
                                (filename[x],
                                 every_docx_name,
                                 temp,
                                 cishu)
                            cishu=cishu+1

                    #temp_content.append(temp)
                # print(len(temp_content))
                #content.append(temp_content)
                # print(every_file_path)

            x=x+1



            #print(files)


    finally:
        print("文件读取完成")
        #return filename,all_file,docx_name
'''文件操作结束，开始es'''
'''文件操作结束，开始es'''
'''文件操作结束，开始es'''

class ElasticObj:
    def __init__(self, index_name,index_type,ip ="localhost"):

        self.index_name =index_name
        self.index_type = index_type
        # 无用户名密码状态
        #self.es = Elasticsearch([ip])
        #用户名密码状态
        self.es = Elasticsearch([ip],
                                http_auth=('elastic',
                                           'password'),
                                port=9200)

    def create_index(self,index_name="ott",index_type="ott_type"):
        '''
        创建索引,创建索引名称为ott，类型为ott_type的索引
        :param ex: Elasticsearch对象
        :return:
        '''
        #创建映射
        _index_mappings = {
            "mappings": {
                self.index_type: {
                    "properties": {
                        "source": {
                            "type": "text",
                            "index": True,
                            "analyzer": "ik_max_word",
                            "search_analyzer": "ik_max_word"
                        }
                    }
                }

            }
        }
        if self.es.indices.exists\
                    (index=self.index_name) is not True:
            res = self.es.indices.create\
                (index=self.index_name, body=_index_mappings)
            print('1',res)
    def ReadFile(self,filepath):
        # filepath='C:\\Users\\wy\\Desktop\\data' \
        #          '\\elasticsearch\\data.txt'
        if os.path.exists(filepath) and os.path.isfile(filepath):
            print("*********文件成功读取完毕*********")
            with open(filepath) as f:
                temp_list=[]
                for line in f:
                    temp_set={}
                    #print(line)
                    temp_set=str(line)
                    temp_list.append(temp_set)
                #print(temp_list)
            return temp_list
        else:
            print("错误：文件目录不存在或文件不存在")
    def bulk_Index_Data(self,category,docname,content,cishu):

        if isinstance(content,str):   #判断是否为表类型
            #print('插入数据')
           # for category,name,line in namelist,docname,filelist:

            action = {
                    'category': category,
                    'name': docname,
                    "title": content}
            #ACTIONS.append(action)
            self.es.index(index=self.index_name,
                          doc_type=self.index_type,
                          body=action)
            print('已经插入',cishu,'条记录')
        else:
            print("错误：文件尚未成功从目录中写入库")

    def Search_data(self,input_text):
        # doc = {'query': {'match_all': {}}}
        #start_time = time()
        doc = {
            "query": {
                "match":{
                    "title": {

                            "query": input_text,
                            "operator": "and"

                    }
                }
            }
        }
        _searched = self.es.search(
            index=self.index_name,
            doc_type=self.index_type,
            body=doc)
        i=0
        last_category = []
        last_docxname = []
        last_sentence = []
        for hit in _searched['hits']['hits']:
            #print (hit['_source'])
            #print ( hit['_source']['title'])
            # print(hit['_score'])
            last_category.append(hit['_source']['category'])
            last_docxname.append(hit['_source']['name'])
            last_sentence.append(hit['_source']['title'])
            i=i+1
            if i==1:
                break
        #print(len(last_sentence))
        for temp in last_sentence:
            print(temp)
        print(last_category)
        print(last_docxname)
        #cost_time = time()-start_time
        #print(cost_time)


if __name__ == '__main__':
    obj = ElasticObj("3", "ott_type", ip="localhost")
    obj.create_index()
    readfile()
    # # #return filename, all_file, docx_name
    # # docname=readfile()[2]
    # # filelist=readfile()[1]
    # # namelist=readfile()[0]
    while(1):
        print("请输入与要匹配的字符串,输入 0 终止查询")
        input_text = input()
        if input_text != str(0):
            print('查询结果如下')
            obj.Search_data(input_text=input_text)
        else:
            print("查询终止")
            break
