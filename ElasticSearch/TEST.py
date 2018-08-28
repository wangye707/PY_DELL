#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : TEST.py
# @Author: WangYe
# @Date  : 2018/8/15
# @Software: PyCharm
from datetime import datetime
from elasticsearch import Elasticsearch
es=Elasticsearch()
#
# class test:
#     #def __init__(self, index_name, index_type, ip="127.0.0.1"):
#     def __init__(self, index_name, index_type):
#         '''
#
#         :param index_name: 索引名称
#         :param index_type: 索引类型
#         '''
#         self.index_name = index_name
#         self.index_type = index_type
#         # 无用户名密码状态
#         # self.es = Elasticsearch([ip])
#         # 用户名密码状态
#         #self.es = Elasticsearch([ip], http_auth=('elastic', 'password'), port=9200)
#
#
#     def create_index(self, index_name="ott", index_type="ott_type"):
#         '''
#         创建索引,创建索引名称为ott，类型为ott_type的索引
#         :param ex: Elasticsearch对象
#         :return:
#         '''
#         # 创建映射
#         _index_mappings = {
#             "mappings": {
#                 self.index_type: {
#                     "properties": {
#                         "title": {
#                             "type": "text",
#                             "index": True,
#                             "analyzer": "ik_max_word",
#                             "search_analyzer": "ik_max_word"
#                         },
#                         "date": {
#                             "type": "text",
#                             "index": True
#                         },
#                         "keyword": {
#                             "type": "string",
#                             "index": "not_analyzed"
#                         },
#                         "source": {
#                             "type": "string",
#                             "index": "not_analyzed"
#                         },
#                         "link": {
#                             "type": "string",
#                             "index": "not_analyzed"
#                         }
#                     }
#                 }
#
#             }
#         }
#         # if self.es.indices.exists(index=self.index_name) \
#         #         is not True:
#         #     res = self.es.indices.create(index=self.index_name,
#         #                                  body=_index_mappings)
#         #     print(res)
#     def bulk_Index_Data(self):
#         '''
#         用bulk将批量数据存储到es
#         :return:
#         '''
#         list = [
#             {"date": "2017-09-13",
#              "source": "慧聪网",
#              "link": "http://info.broadcast.hc360.com/2017"
#                      "/09/130859749974.shtml",
#              "keyword": "电视",
#              "title": "付费 电视 行业面临的转型和挑战"
#              },
#             {"date": "2017-09-13",
#              "source": "中国文明网",
#              "link": "http://www.wenming.cn/xj_pd/yw/"
#                      "201709/t20170913_4421323.shtml",
#              "keyword": "电视",
#              "title": "电视 专题片《巡视利剑》广获好评：铁腕反腐凝聚党"
#                       "心民心"
#              },
#             {"date": "2017-09-13",
#              "source": "人民电视",
#              "link": "http://tv.people.com.cn/BIG5/n1/2017/0913/"
#                      "c67816-29533981.html",
#              "keyword": "电视",
#              "title": "中国第21批赴刚果（金）维和部隊启程--人民 "
#                       "电视 --人民网"
#              },
#             {"date": "2017-09-13",
#              "source": "站长之家",
#              "link": "http://www.chinaz.com/news/2017/0913/"
#                      "804263.shtml",
#              "keyword": "电视",
#              "title": "电视 盒子 哪个牌子好？ 吐血奉献三大选购秘笈"
#              }
#         ]
#         ACTIONS = []
#         i = 1
#         for line in list:
#             action = {
#                 "_index": self.index_name,
#                 "_type": self.index_type,
#                 "_id": i, #_id 也可以默认生成，不赋值
#                 "_source": {
#                     "date": line['date'],
#                     "source": line['source'],
#                     "link": line['link'],
#                     "keyword": line['keyword'],
#                     "title": line['title']}
#             }
#             i += 1
#             ACTIONS.append(action)
#             # 批量处理
        #success, _ = bulk(self.es, ACTIONS, index=self.index_name, raise_on_error=True)
       # print('Performed %d actions' % success)
    # def Get_Data_By_Body(self):
    #         # doc = {'query': {'match_all': {}}}
    #     doc = {
    #         "query": {
    #             "match": {
    #                     "keyword": "电视"
    #             }
    #         }
    #     }
    #    # _searched = self.es.search(index=self.index_name,
    #                   #             doc_type=self.index_type, body=doc)
    #
    #     for hit in _searched['hits']['hits']:
    #             # print hit['_source']
    #         print (hit['_source']['date'], hit['_source']['source'],
    #                hit['_source']['link'], hit['_source']['keyword'],
    #                hit['_source']['title'])
#
doc={
    'autor': '中华',
    'text' : '中华人民共和国国歌',
    "search_analyzer": "ik_max_word",
    'timestamp':datetime.now(),
}

res=es.index(index='wy',doc_type='test',id=1,body=doc)
print(res['result'])

res = es.get(index="wy", doc_type='test', id=1)
print(res['_source'])
'''fengexian'''
# test=test('name','type')
# test.create_index()
# test.bulk_Index_Data()
#test.Get_Data_By_Body()
#res=es.get(index="name", doc_type='type', id=1)