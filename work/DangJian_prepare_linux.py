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
#import win32com.client
PATH = os.path.abspath(os.path.dirname(sys.argv[0]))
#print(PATH)
PATH_DATA=r'./DangJian_prepare'
# 主要执行函数
'''将docx转化为txt'''

'''遍历TXT文件并且调用es插入数据'''


def readfile():
    docx_name=[]   #存储文件名
    filename=[]   #存储文件种类名
    all_file=[]    #存储所有文件
    #filename_fenci=[]
    #temp1=[]
    print('文件开始读取')
    x=-1
    try:
        for root, dirs, files in os.walk(PATH_DATA):
            #print(x)
            if x==-1:
                #print(dirs)
                for name in dirs:#访问第一层文件夹
                    #print(name)
                    #filename_fenci.append(jieba.lcut(name))
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
                    new_every_docx_name=\
                        every_docx_name.replace('.py','')
                    filename_fenci=jieba.lcut(new_every_docx_name)
                    #分档名分词存储
                    # print(filename_fenci)
                    # print(every_docx_name)
                    every_file_path = \
                        os.path.join(root, every_docx_name)

                    # print(every_file_path)
                    # print(every_file_path)
                    # print(filename)
                    try:#判断编码问题
                        with codecs.open(every_file_path,
                                         encoding='utf-8') as f:
                            temp1 = f.read()


                            temp = eval(temp1[temp1.index('['):])
                            #print(temp1)
                            #print(temp)
                            content = []
                            content_fenci=[]
                            #print(len(temp))
                            ''''''
                            for hang in range(len(temp)):
                                sentence_fenci = jieba.\
                                    lcut(temp[hang])
                                sentence = temp[hang]
                                #print(sentence)
                                content.append(sentence)
                                content_fenci.\
                                    append(sentence_fenci)


                            # print('文件长度', len(temp))
                            '''执行文档插入操作'''
                            # self, category, docname, docname_fenci,
                            # content, content_fenci, cishu
                            wy.Index_Data \
                                (filename[x],
                                    new_every_docx_name,
                                    filename_fenci,
                                    content,
                                    content_fenci,
                                    cishu)

                            cishu = cishu + 1


                    except UnicodeDecodeError:
                        #如果不是gbk，执行utf-8操作
                        with codecs.open(every_file_path,
                                         encoding='gbk') as f:
                            temp1 = f.read()

                            temp = eval(temp1[temp1.index('['):])
                           # print(len(temp))


                            content = []
                            content_fenci=[]

                            ''''''
                            for hang in range(len(temp)):
                                sentence_fenci = jieba.\
                                    lcut(temp[hang])
                                sentence = temp[hang]
                                #print(sentence)
                                content.append(sentence)
                                content_fenci.\
                                    append(sentence_fenci)


                            # print('文件长度', len(temp))
                            '''执行文档插入操作'''
                            # self, category, docname, docname_fenci,
                            # content, content_fenci, cishu
                            # wy.index_name()
                            wy.Index_Data \
                                (filename[x],
                                    every_docx_name,
                                    filename_fenci,
                                    content,
                                    content_fenci,
                                    cishu)

                            cishu = cishu + 1

                    #temp_content.append(temp)
                # print(len(temp_content))
                #content.append(temp_content)
                # print(every_file_path)

            x=x+1



            #print(files)


    finally:
        print("文件读取完成")
'''文件操作结束，开始es'''
'''文件操作结束，开始es'''
'''文件操作结束，开始es'''

class ElasticWy:
    def __init__(self, index_name,index_type,ip ="localhost"):

        self.index_name =index_name
        self.index_type = index_type
        self.es = Elasticsearch([ip],
                                http_auth=('elastic',
                                           'password'),
                                port=9200)

    def create_index(self,index_name,index_type):
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
    def Index_Data(self,category,docname,docname_fenci,
                   content,content_fenci,cishu):

        if isinstance(content,list):   #判断是否为表类型
            #print('插入数据')
           # for category,name,line in namelist,docname,filelist:

            action = {
                    'category': category,
                    'name': docname,
                    'name_fenci': docname_fenci,
                    "title": content,
                    'title_fenci': content_fenci
                    }
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
        last_docxname_fenci=[]
        last_sentence = []
        last_sentence_fenci=[]
        for hit in _searched['hits']['hits']:
            #print (hit['_source'])
            #print ( hit['_source']['title'])
            # print(hit['_score'])
            # action = {
            #     'category': category,
            #     'name': docname,
            #     'name_fenci': docname_fenci,
            #     "title": content,
            #     'title_fenci': content_fenci
            # }
            last_category.append(hit['_source']['category'])
            last_docxname.append(hit['_source']['name'])
            last_docxname_fenci\
                .append(hit['_source']['name_fenci'])
            last_sentence.append(hit['_source']['title'])
            last_sentence_fenci.append(hit['_source']['title_fenci'])
            i=i+1
            if i==2:
                break
        #print(len(last_sentence))
        # for temp in last_sentence:
        #     print('原文',temp)
        # for temp in last_sentence_fenci:
        #     print('分词',temp)
        # print('内容分类',last_category)
        # print('分档名',last_docxname)
        # print('文档名_分词',last_docxname_fenci)
        return last_sentence,last_sentence_fenci,last_category\
            ,last_docxname,last_docxname_fenci
        #cost_time = time()-start_time
        #print(cost_time)
def a(question):
    last_sentence=wy.Search_data(input_text)[0]
    last_sentence_fenci=wy.Search_data(input_text)[1]
    last_category=wy.Search_data(input_text)[2]
    last_docxname=wy.Search_data(input_text)[3]
    last_docxname_fenci=wy.Search_data(input_text)[4]
    print(last_docxname_fenci)
    return last_sentence, last_sentence_fenci, last_category \
        , last_docxname, last_docxname_fenci

if __name__ == '__main__':
    wy = ElasticWy("dj_fgzd", "zeno", ip="139.129.129.77")
    wy.create_index(index_name="dj_fgzd",index_type="zeno")
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
            wy.Search_data(input_text=input_text)
        else:
            print("查询终止")
            break
