#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : test1.py
# @Author: WangYe
# @Date  : 2018/8/15
# @Software: PyCharm
#coding:utf8
import os
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch.helpers import bulk
import elasticsearch
import json
from elasticsearch_dsl import DocType, Text, Keyword
from elasticsearch_dsl.analysis import CustomAnalyzer as _CustomAnalyzer
from elasticsearch_dsl.analysis import CustomTokenFilter
from elasticsearch_dsl.connections import connections
connections.create_connection(hosts=["localhost"])
es = Elasticsearch()
'''读取文件'''
path='C:\\Users\\wy\\Desktop\\data\\elasticsearch'
class Read():
    '''读取单个文件'''
    def readFile(self, file, s):
        f1 = open(file, "r")
        det = Detail()
        try:
            lines = f1.readlines()
            det.set_date(es, lines, s)
        finally:
            f1.close()
        return lines
    '''遍历目录下的文件'''
    def eachFile(self, filepath):
        pathDir = os.listdir(filepath)  # 获取当前路径下的文件名，返回List
        for s in pathDir:
            newDir = os.path.join(filepath, s)  # 将文件名加入到当前文件路径后面
            if os.path.isfile(newDir):  # 如果是文件
                if os.path.splitext(newDir)[1] == ".txt":  # 判断是否是txt
                    s = s.split(".")
                    self.readFile(newDir, s)  # 读文件
                else:
                    self.eachFile(newDir)
'''参数设置'''
class CustomAnalyzer(_CustomAnalyzer):
    def get_analysis_definition(self):
        return {}
#synonyms_path = r"F:\sympathy\synonym.txt"   #同义词库路径
# myfilter = CustomTokenFilter("synonym",
#                              synonyms_path=synonyms_path) #
myfilter = CustomTokenFilter("synonym") #
ik_analyzer = CustomAnalyzer("ik_max_word")
'''创建连接'''
class ElasticSearchClient(object):
    @staticmethod
    def get_es_servers():
        es_servers = [{
            "host": "localhost",
            "port": "9200"
        }]
        es_client = elasticsearch.Elasticsearch(hosts=es_servers)
        return es_client
'''写入数据的格式'''
class LoadElasticSearch(object):
    es_client =ElasticSearchClient().get_es_servers()
    def __init__(self):
        self.index = "roger"
        self.doc_type ="test"
        #self.es_client = ElasticSearchClient.get_es_servers()
        self.set_mapping()

    def set_mapping(self):
        """
        设置mapping
        """
        chinese_field_config = {
            "type": "string",
            #"store": "no",
           # "term_vector": "with_positions_offsets",
            "analyzer": "ik_max_word",
            "search_analyzer": "ik_max_word",
            "include_in_all": "true",
            "boost": 8
        }

        mapping = {
            self.doc_type: {
                "_all": {"enabled": False},

                "properties": {
                    "document_id": {
                        "type": "integer"
                    },
                    "content": chinese_field_config
                }
            }
        }
        #
        if not self.es_client.indices.exists(index=self.index):
            # 创建Index和mapping
            self.es_client.indices.create\
                (index=self.index, ignore=400)
            self.es_client.indices.put_mapping\
                (index=self.index, doc_type=self.doc_type,
                 body=mapping)



'''写入数据格式'''
class Detail():
    # red = Read()
    def set_date(self, es,
                 line_list, s,
                 index_name="roger",
                 doc_type_name="test"):
        # 读入数据
        fileds = []
        for line in line_list:

            attr = json.loads(line)
            fileds.append(attr["value"])
        # 创建ACTIONS
        ACTIONS = []
        action = {
            "_index": index_name,
            "_type": doc_type_name,
            "_source": {
                "user_id": s[0],
                "criterion": fileds[0],
                "intellectual_property": fileds[1],
                "paper": fileds[2],
                "research_project": fileds[3],
                "professional_certificate": fileds[4],
                "academic_activities": fileds[5],
                "experience": fileds[6],
                "further_study": fileds[7],
                "personal_register": fileds[8],
                "expert_title": fileds[9],
                "research": fileds[10],
                "domestic_studies": fileds[11],
                "professional_qualification": fileds[12],
            }
        }
        ACTIONS.append(action)

            # 批量处理
        success, _ = bulk(es, ACTIONS,
                          index=index_name,
                          raise_on_error=True)
        print('Performed %d actions' % success)
'''查询相应消息'''
# es = connections.create_connection(MyType._doc_type.using)
# class SearchEdit():
#     def Search(self, row_obj,
#                index_name="roger", doc_type="test"):
#         q = {
#             "query": {
#                 "match": {
#                     "user_id": row_obj
#                 }
#             }
#         }
#         result = es.search(index_name, doc_type, q)
#         r = result["hits"]["total"]
#         if r > 0:
#             return 1
#         else:
#             return 0
#
#     def detailSearch(self, row_obj,
#                      index_name="roger", doc_type="test"):
#         q = {
#             "query": {
#                 "multi_match": {
#                     "query": row_obj,
#                     "fields":
#                         ["user_id",
#                          "criterion",
#                          "intellectual_property",
#                          "paper", "research_project",
#                          "professional_certificate",
#                          "academic_activities",
#                          "experience", "further_study",
#                          "personal_register",
#                          "expert_title",
#                          "research",
#                          "domestic_studies",
#                          "professional_qualification"],
#                     "analyzer": ik_analyzer
#                 }
#             }
#         }
#         result = es.search(index_name, doc_type, q)
#         hit_list = []
#         # print(result)
#         for hit in result["hits"]["hits"]:
#             hit_dict = {}
#             for filed in hit["_source"]:
#                 if hit["_source"][filed] == str(row_obj):
#                     hit_dict["table"] = filed
#             hit_dict["user_id"] = hit["_source"]["user_id"]
#             hit_list.append(hit_dict)
#         print(hit_list)
#         return hit_list
'''查询删除操作'''
def search1():

    doc = {
        'autor': '申燚',
        'text': 'Elasticsearch:cool. bonsai cool.',
        'timestamp': datetime.now(),
    }
    doc1 = {
        'name': 'lucy',
        'sex': 'female',
        'age': 10,
    }
    doc3 = {
        'name': 'liubei',
        'sex': 'male',
        'age': 30,
    }
    doc2 = {
        'autor': 'sihuo',
        'text': 'hello',
        'timestamp': datetime.now(),
    }
    # res=es.index(index='text-index',doc_type='tweet',id=1,body=doc3)
    # print(es.get(index='text-index',doc_type='tweet',id=1))
    # for i in range(4):
    #     es.delete(index='hz',doc_type='text',id=i)

    # es.update(index='text-index',doc_type='tweet',id=1,body=doc)
    # print(es.get(index='text-index',doc_type='tweet',id=1))

    # 批量操作
    # 1.条件查询
    query = {'query': {'match_all': {}}}
    query1 = {'query': {'term': {'name': 'lucy'}}}
    query2 = {'query': {'range': {'age': {'gt': 8}}}}
    allDoc = es.search(index='roger', doc_type='test', body=query)
    for i in allDoc['hits']['hits']:
        print(i)

    # 2条件删除
    # for i in range(1,6):
    #     query1={'query':{'match':{'user_id':str(i)}}}
    #     #query2={'query':{'range':{'user_id':{'lt':10}}}}
    #     es.delete_by_query(index='roger',body=query1,doc_type='test')

    # 批量插入
    doc4 = [
        {"index": {}},
        {'name': 'sunshangxiang', 'age': 18, 'sex': 'female', 'address': u'北京'},
        {"index": {}},
        {'name': 'caocao', 'age': 25, 'sex': 'male', 'address': u'上海'},
        {"index": {}},
        {'name': 'guanyu', 'age': 26, 'sex': 'male', 'address': u'广州'},
        {"index": {}},
        {'name': 'zhangfei', 'age': 27, 'sex': 'male', 'address': u'深圳'},
    ]

#es.bulk(index='text-index',doc_type='tweet',body=doc4)
if __name__ == '__main__':
    #MyType.init()
    LoadElasticSearch().set_mapping()
    sentence=Read().eachFile(filepath=path)
    #Detail().set_date(es=es,line_list=sentence,s=)
    #SearchEdit.detailSearch(row_obj='啊')
    search1()
