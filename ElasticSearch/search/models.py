from elasticsearch_dsl import DocType, Text, Keyword
from elasticsearch_dsl.analysis import CustomAnalyzer as _CustomAnalyzer
from elasticsearch_dsl.analysis import CustomTokenFilter
from elasticsearch_dsl.connections import connections
connections.create_connection(hosts=["localhost"])


class CustomAnalyzer(_CustomAnalyzer):
    def get_analysis_definition(self):
        return {}
synonyms_path = r"F:\sympathy\synonym.txt"
myfilter = CustomTokenFilter("synonym", synonyms_path=synonyms_path) #
ik_analyzer = CustomAnalyzer("ik_max_word", filter=myfilter)

class MyType(DocType):
    # 伯乐在线文章类型
    user_id = Keyword()
    criterion = Text(analyzer=ik_analyzer, similarity="BM25") #同义词
    intellectual_property = Text(analyzer="ik_max_word", similarity="BM25")
    paper = Text(analyzer="ik_max_word", similarity="BM25")
    research_project = Text(analyzer="ik_max_word", similarity="BM25")
    professional_certificate = Text(analyzer="ik_max_word", similarity="BM25")
    academic_activities = Text(analyzer="ik_max_word", similarity="BM25")
    experience = Text(analyzer="ik_max_word", similarity="BM25")
    further_study = Text(analyzer="ik_max_word", similarity="BM25")
    personal_register = Text(analyzer="ik_max_word", similarity="BM25")
    expert_title = Text(analyzer="ik_max_word", similarity="BM25")
    research = Text(analyzer="ik_max_word", similarity="BM25")
    domestic_studies = Text(analyzer="ik_max_word", similarity="BM25")
    professional_qualification = Text(analyzer="ik_max_word", similarity="BM25")

    class Meta:
        index = "roger"
        doc_type = "test"

if __name__ == "__main__":
   MyType.init()
