import elasticsearch

class ElasticSearchClient(object):
    @staticmethod
    def get_es_servers():
        es_servers=[{
            'host':'localhost',
            'port':'9200'
        }]
        es_client=elasticsearch.Elasticsearch(hosts=es_servers)
        return es_client

class LoadElasticSearch(object):
    def __init__(self):
        self.index='hz'
        self.doc_type='text'
        self.es_client=ElasticSearchClient.get_es_servers()
        self.set_mapping()
    def set_mapping(self):
        '''设置mapping'''
        chinese_field_config={
            'type':'string',
            'store':'no',
            'term_vector':'with_positions_offsets',
            'analyzer':'ik_smart',
            'search_analyzer':'ik_smart',
            'include_in_all':'true',
            'boost':8
        }

        mapping={
            self.doc_type:{
                '_all':{'enabled':False},
                'properties':{
                    'document_id':{
                        'type':'integer'
                    },
                    'content':chinese_field_config
                }
            }
        }
        if not self.es_client.indices.exists(index=self.index):
            self.es_client.indices.create(index=self.index,
                                          ignore=400)
            self.es_client.indices.\
                put_mapping(index=self.index,
                            doc_type=self.doc_type, body=mapping)

    def add_date(self, row_obj):
        """
        单条插入ES
        """
        _id = row_obj.get("_id", 1)
        row_obj.pop("_id")
        self.es_client.index\
            (index=self.index, doc_type=self.doc_type,
             body=row_obj, id=_id)

if __name__ == '__main__':

    content_ls = [
                u"美国留给伊拉克的是个烂摊子吗",
                u"公安部：各地校车将享最高路权",
                u"中韩渔警冲突调查：韩警平均每天扣1艘中国渔船",
                u"中国驻洛杉矶领事馆遭亚裔男子枪击 嫌犯已自首"
            ]

    load_es = LoadElasticSearch()
            # 插入单条数据测试
    for index, content in enumerate(content_ls):
        write_obj = {
                    "_id": index,
                    "document_id": index,
                    "content": content
                }
        load_es.add_date(write_obj)
