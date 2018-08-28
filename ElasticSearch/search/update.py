from ElasticSearch.search.models import MyType
from elasticsearch_dsl.connections import connections

es = connections.create_connection(MyType._doc_type.using)


class Update():
    def update(self,
               row_obj,
               index_name="roger",
               doc_type_name="test", ):
        updateBody = {
            "query": {
                "match": {
                    "user_id": row_obj
                }
            }
        }
        # es.update_by_query(index_name, doc_type_name, updateBody)
        es.delete_by_query(index_name,
                           updateBody,
                           doc_type_name)
