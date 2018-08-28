#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : ElasticSearch_TEST.py
# @Author: WangYe
# @Date  : 2018/8/20
# @Software: PyCharm
#!D:/workplace/python
import os
from threading import Thread
import time
from os import walk
#import CSVOP
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from time import time

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
                        "title": {
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
    def bulk_Index_Data(self,filelist):
        '''
        用bulk将批量数据存储到es
        :return:
        '''
        ACTIONS = []
        i = 1
        #a=filelist
        #print(type(filelist))
        if isinstance(filelist,list):   #判断是否为表类型
            print('*********文件成功从目录中写入库*********')
            for line in filelist:
                action = {
                    "_index": self.index_name,
                    "_type": self.index_type,
                    "_id": i, #_id 也可以默认生成，不赋值
                    "_source": {
                        "title": line}
                }
                i += 1
                ACTIONS.append(action)
                # 批量处理
            success, _ = bulk(self.es,
                              ACTIONS, index=self.index_name,
                              raise_on_error=True)
            print('一共执行 %d 条记录' % success)
        else:
            print("错误：文件尚未成功从目录中写入库")

    def Search_data(self,input_text):
        # doc = {'query': {'match_all': {}}}
        start_time = time()
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
        last_sentence = []
        for hit in _searched['hits']['hits']:
            #print (hit['_source'])
            #print ( hit['_source']['title'])
            # print(hit['_score'])
            last_sentence.append(hit['_source']['title'])
            i=i+1
            if i==10:
                break
        #print(len(last_sentence))
        for temp in last_sentence:
            print(temp)
        cost_time = time()-start_time
        print(cost_time)
if __name__ == '__main__':
    # thread_pool = []
    # question_pool = ['人阶级的先锋',
    #                  '会以来，以邓小平同志为主要代',
    #                  '来，以习近平同志为主要代表的中国共产党人，顺',
    #                  '中国共产党在社会主义初级阶段的基本路线是',
    #                  '中，必须坚持以经济建设',
    #                  '共产党领导人民发展社会主义市场经济。',
    #                  '建社会主义和谐社会。按照民主',
    #                  '产党维护和发展平等团结互助和谐的社会主义民',
    #                  '坚持解放思想，实事求是，与时俱进，求真务实。',
    #                  '坚持全心全意为人民服务。党除了工人阶级和最广大人',
    #                  '人阶级的先锋',
    #                  '会以来，以邓小平同志为主要代',
    #                  '来，以习近平同志为主要代表的中国共产党人，顺',
    #                  '中国共产党在社会主义初级阶段的基本路线是',
    #                  '中，必须坚持以经济建设',
    #                  '共产党领导人民发展社会主义市场经济。',
    #                  '建社会主义和谐社会。按照民主',
    #                  '产党维护和发展平等团结互助和谐的社会主义民',
    #                  '坚持解放思想，实事求是，与时俱进，求真务实。',
    #                  '坚持全心全意为人民服务。党除了工人阶级和最广大人']*5
    obj = ElasticObj("1","ott_type",ip ="localhost")
    # pool_size = 100
    # for i in range(pool_size):
    #     t = Thread(target=obj.Search_data, args=(question_pool[i] ,))
    #     thread_pool.append(t)
    # for i in range(pool_size):
    #     thread_pool[i].run()
    #t1=td.Thread(target=)
    filepath = 'C:\\Users\\wy\\Desktop\\data' \
               '\\elasticsearch\\data.txt'
    # #filepath='ads'
    sentence_list=obj.ReadFile(filepath=filepath)
    obj.create_index()
    #
    obj.bulk_Index_Data(sentence_list)
    while 1:
        print("请输入与要匹配的字符串,输入 0 终止查询")
        input_text=input()
        if input_text!=str(0):
            print('查询结果如下')
            obj.Search_data(input_text=input_text)
        else:
            print("查询终止")
            break