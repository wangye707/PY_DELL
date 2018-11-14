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


def readfile(path):
    docx_name=[]   #存储文件名
    filename=[]   #存储文件种类名
    all_file=[]    #存储所有文件
    #filename_fenci=[]
    #temp1=[]
    PATH_DATA=path
    print('文件开始读取')
    x=-1
    id=1
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
                                    cishu,
                                    str(id)
                                    )
                            id = id + 1

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
                                    cishu,
                                    str(id))
                            id = id +1
                            cishu = cishu + 1

                    #temp_content.append(temp)
                # print(len(temp_content))
                #content.append(temp_content)
                # print(every_file_path)

            x=x+1



            #print(files)


    finally:
        print("文件读取完成")
#readfile()
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
                   content,content_fenci,cishu,id):

        if content!=None:   #判断是否为表类型
            #print('插入数据')
           # for category,name,line in namelist,docname,filelist:

            action = {
                # # "_index": self.index_name,
                # # "_type": self.index_type,
                # # #"_id": id,  # _id 也可以默认生成，不赋值
                # "source":
                #     {
                        'category': category,
                        'name': docname,
                        'name_fenci': docname_fenci,
                        "title": content,
                        'title_fenci': content_fenci,
                        "id": id,  # _id
                        }
           # }
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
        last_id=[]
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
            last_id.append(hit['_id'])
            i=i+1
            if i==2:
                break
        #print(len(last_sentence))
        for temp in last_sentence:
            print('原文',temp)
        for temp in last_sentence_fenci:
            print('分词',temp)
        print('内容分类',last_category)
        print('分档名',last_docxname)
        print('文档名_分词',last_docxname_fenci)
        print('delete_key',last_id)
        #cost_time = time()-start_time
        #print(cost_time)
    def delete_content(self,index_name,doc_type,key):
        print('准备删除操作')
        try:
            wy.es.delete(index_name, doc_type, id=key)
            #print('删除成功')
        except :
            print('我猜您输入的 delete_key 很可能不存在，请根据（ 1.查询 ）中反馈的 delete_key 进行删除操作')
        else:
            print('删除成功')



if __name__ == '__main__':
    index_name='dj_fgzd'
    doc_type="zeno"
    ip='139.129.129.77'
    wy = ElasticWy(index_name, doc_type, ip)
    PATH_DATA = r'C:\Users\wy\Desktop\Data_Test' #数据插入目录
    PATH_update=r'C:\Users\wy\Desktop\Insert'  #更新数据路径
    while(1):
        print("请输入需要执行的操作：")
        print('1.查询')
        print('2.删除')
        print('3.插入')
        print('4.更新')
        print('0.退出')
        input_num = input()
        if input_num == str(1):
        #print("请输入需要执行的操作：")
            while(1):
                print("请输入与要匹配的字符串,输入 0 返回")
                input_text = input()
                if input_text != str(0):
                    print('查询结果如下')
                    wy.Search_data(input_text=input_text)
                else:
                    print("查询终止")
                    break
        # '''删除'''
        if input_num == str(2):
            while(1):
                print("请输入与要删除的ID,输入 0 返回")
                input_text = input()
                if input_text != str(0):
                    #print('查询结果如下')
                    wy.delete_content(index_name, doc_type,input_text)
                    #wy.Search_data(input_text=input_text)
                else:
                    print("删除终止")
                    break
        if input_num == str(3):
            print('插入操作将会插入所有库中的文件，插入各个参数为：')
            print('PATH_DATA：',PATH_DATA)
            print('index_name:',index_name)
            print('doc_type:',doc_type)
            print('ip:',ip)
            print('输入 1 继续,按其他任意键返回上一层')
            str_select=input()
            if str_select == str(1):
                print('开始执行插入操作')
                wy.create_index(index_name=index_name, index_type=doc_type)
                readfile(PATH_DATA)
                print('插入操作执行完成')
            else:
                continue
        if input_num == str(4):
            print('开始读入更新路径文件夹')
            try:
                readfile(PATH_update)
            except FileNotFoundError:
                print("路径未读取到，"
                      "请将更新文件放入'C:\\Users\\wy\\Desktop\\Insert'下’,如："
                      "'C:\\Users\wy\\Desktop\\Insert\\0新名词\\百科.py'")
            except ConnectionRefusedError:
                print('ElasticSearch 链接出现问题')
            except:
                print('未知错误')
            else:
                print('文件更新成功')
        if input_num == str(0):
            break
        # if input_num != str(0) or str(1) or str(2) or str(3):
        #     print('你他妈是傻逼吗？看不清楚输入是几？看清楚输入数字！猪！重输')
        #     continue