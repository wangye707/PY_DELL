# -*- coding: utf-8 -*-

import elasticsearch


class ElasticSearchClient(object):
    @staticmethod
    def get_es_servers():
        es_servers = [{
            "host": "localhost",
            "port": "9200"
        }]
        es_client = elasticsearch.Elasticsearch(hosts=es_servers)
        return es_client


class SearchData(object):
    index = 'hz'
    doc_type = 'text'

    @classmethod
    def search(cls, field, query, search_offset, search_size):
        # 设置查询条件 content,中国，0，30
        es_search_options = cls.set_search_optional(field, query)
        # 发起检索。
        es_result = cls.get_search_result(es_search_options, search_offset, search_size)
        # 对每个结果, 进行封装。得到最终结果
        final_result = cls.get_highlight_result_list(es_result, field)
        return final_result

    @classmethod
    def get_highlight_result_list(cls, es_result, field):
        result_items = es_result['hits']['hits']
        final_result = []
        for item in result_items:
            item['_source'][field] = item['highlight'][field][0]
            final_result.append(item['_source'])
        return final_result

    @classmethod
    def get_search_result(cls, es_search_options, search_offset, search_size):
        es_result = ElasticSearchClient.get_es_servers().search(
            index=cls.index,
            doc_type=cls.doc_type,
            body=es_search_options,
            from_=search_offset,
            size=search_size
        )
        return es_result

    @classmethod
    def set_search_optional(cls, field, query):
        es_search_options = {
            "query": {
                "match": {
                    field: {
                        "query": query,
                        "slop": 10
                    }
                }
            },
            "highlight": {
                "fields": {
                    "*": {
                        "require_field_match": True,
                    }
                }
            }
        }
        return es_search_options


if __name__ == '__main__':
    final_results = SearchData().search("content", "中国", 0, 30)
    for obj in final_results:
        for k, v in obj.items():
            print(k, ":", v)
        print("=======")