from elasticsearch_dsl.connections import connections
from ElasticSearch.search.models import MyType
from elasticsearch_dsl.analysis import \
    CustomAnalyzer as _CustomAnalyzer
from elasticsearch_dsl.analysis import CustomTokenFilter


es = connections.create_connection(MyType._doc_type.using)


class CustomAnalyzer(_CustomAnalyzer):
    def get_analysis_definition(self):
        return {}

synonyms_path = r"C:\Users\wy\Desktop\data\
elasticsearch\1.txt"
myfilter = CustomTokenFilter("synonym",
                             synonyms_path=synonyms_path)
ik_analyzer = CustomAnalyzer("ik_max_word", filter=myfilter)


class SearchEdit():
    def Search(self, row_obj,
               index_name="roger", doc_type="test"):
        q = {
            "query": {
                "match": {
                    "user_id": row_obj
                }
            }
        }
        result = es.search(index_name, doc_type, q)
        r = result["hits"]["total"]
        if r > 0:
            return 1
        else:
            return 0

    def detailSearch(self, row_obj,
                     index_name="roger", doc_type="test"):
        q = {
            "query": {
                "multi_match": {
                    "query": row_obj,
                    "fields":
                        ["user_id",
                         "criterion",
                         "intellectual_property",
                         "paper", "research_project",
                         "professional_certificate",
                         "academic_activities",
                         "experience", "further_study",
                         "personal_register",
                         "expert_title",
                         "research",
                         "domestic_studies",
                         "professional_qualification"],
                    "analyzer": ik_analyzer
                }
            }
        }
        result = es.search(index_name, doc_type, q)
        hit_list = []
        # print(result)
        for hit in result["hits"]["hits"]:
            hit_dict = {}
            for filed in hit["_source"]:
                if hit["_source"][filed] == str(row_obj):
                    hit_dict["table"] = filed
            hit_dict["user_id"] = hit["_source"]["user_id"]
            # hit_dict["criterion"] = hit["_source"]["criterion"]
            # hit_dict["intellectual_property"] = hit["_source"]["intellectual_property"]
            # hit_dict["paper"] = hit["_source"]["paper"]
            # hit_dict["research_project"] = hit["_source"]["research_project"]
            # hit_dict["professional_certificate"] = hit["_source"]["professional_certificate"]
            # hit_dict["academic_activities"] = hit["_source"]["academic_activities"]
            # hit_dict["experience"] = hit["_source"]["experience"]
            # hit_dict["further_study"] = hit["_source"]["further_study"]
            # hit_dict["personal_register"] = hit["_source"]["personal_register"]
            # hit_dict["expert_title"] = hit["_source"]["expert_title"]
            # hit_dict["research"] = hit["_source"]["research"]
            # hit_dict["domestic_studies"] = hit["_source"]["research"]
            # hit_dict["professional_qualification"] = hit["_source"]["professional_qualification"]

            hit_list.append(hit_dict)
        print(hit_list)
        return hit_list