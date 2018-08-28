# coding=utf-8
from elasticsearch.helpers import bulk
import json

# sys.setdefaultencoding('utf-8')


# 设置mapping
# def set_mapping(es, index_name="roger_search", doc_type_name="user"):
#     my_mapping = {
#         "en": {
#             "properties": {
#                 "user_id": {
#                     "type": "keyword",
#                 },
#                 "criterion": {
#                     "type": "text",
#                     "analyzer": "ik_max_word",
#                     "similarity": "BM25"
#                 },
#                 "intellectual_property": {
#                     "type": "text",
#                     "analyzer": "ik_max_word",
#                     "similarity": "BM25"
#                 },
#                 "paper": {
#                     "type": "text",
#                     "analyzer": "ik_max_word",
#                     "similarity": "BM25"
#                 },
#                 "research_project": {
#                     "type": "text",
#                     "analyzer": "ik_max_word",
#                     "similarity": "BM25"
#                 },
#                 "professional_certificate": {
#                     "type": "text",
#                     "analyzer": "ik_max_word",
#                     "similarity": "BM25"
#                 },
#                 "academic_activities": {
#                     "type": "text",
#                     "analyzer": "ik_max_word",
#                     "similarity": "BM25"
#                 },
#                 "experience": {
#                     "type": "text",
#                     "analyzer": "ik_max_word",
#                     "similarity": "BM25"
#                 },
#                 "further_study": {
#                     "type": "text",
#                     "analyzer": "ik_max_word",
#                     "similarity": "BM25"
#                 },
#                 "personal_register": {
#                     "type": "text",
#                     "analyzer": "ik_max_word",
#                     "similarity": "BM25"
#                 },
#                 "expert_title": {
#                     "type": "text",
#                     "analyzer": "ik_max_word",
#                     "similarity": "BM25"
#                 },
#                 "research": {
#                     "type": "text",
#                     "analyzer": "ik_max_word",
#                     "similarity": "BM25"
#                 },
#                 "domestic_studies": {
#                     "type": "text",
#                     "analyzer": "ik_max_word",
#                     "similarity": "BM25"
#                 },
#                 "professional_qualification": {
#                     "type": "text",
#                     "analyzer": "ik_max_word",
#                     "similarity": "BM25"
#                 }
#             }
#         }
#     }
#
#     # 创建Index和mapping
#     create_index = es.indices.create(index=index_name, body=my_mapping)  # {u'acknowledged': True}
#     mapping_index = es.indices.put_mapping(index=index_name, doc_type=doc_type_name,
#                                            body=my_mapping)  # {u'acknowledged': True}
#     if create_index["acknowledged"] != True or mapping_index["acknowledged"] != True:
#         print("Index creation failed...")

# 将文件中的数据存储到es中


class Detail():
    # red = Read()
    def set_date(self, es,
                 line_list,#存取数据的列表
                 s,#user_id
                 index_name="roger",#库名
                 doc_type_name="test"):#种类名
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
    #
    #
    # # 读取参数
    # # def read_args(self):
    # #     parser = argparse.ArgumentParser(description="Search Elastic Engine")
    # #     parser.add_argument("-i", dest="input_file", action="store", help="input file1", required=True)
    # #     # parser.add_argument("-o", dest="output_file", action="store", help="output file", required=True)
    # #     return parser.parse_args()
    #
    #
    # if __name__ == '__main__':
    #     # args = read_args()
    #     # es = Elasticsearch(hosts=["127.0.0.1:9100"], timeout=5000)
    #     # set_mapping(es)
    #     set_date(es)