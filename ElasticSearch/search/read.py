import os
import json
from ElasticSearch.search.detail import Detail
from ElasticSearch.search.search import SearchEdit
from ElasticSearch.search.update import Update
from elasticsearch_dsl.connections import connections
from ElasticSearch.search.models import MyType
es = connections.create_connection(MyType._doc_type.using)

class Read():
    def readFile(self, file, s):
        f1 = open(file, "r")
        det = Detail()
        try:
            lines = f1.readlines()
            det.set_date(es, lines, s)
        finally:
            f1.close()
        return lines

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

    def readEditFile(self, file, s):
        f1 = open(file, "r")
        det = Detail()
        se = SearchEdit
        try:
            lines = f1.readlines()
            if se.Search(s[0]):
                up = Update()
                up.update(s[0])
                det.set_date(es, lines, s)
            else:
                det.set_date(es, lines, s)
        finally:
            f1.close()
        return lines

    def eachFile_0(self, filepath):
        pathDir = os.listdir(filepath)  # 获取当前路径下的文件名，返回List
        for s in pathDir:
            newDir = os.path.join(filepath, s)  # 将文件名加入到当前文件路径后面
            if os.path.isfile(newDir):  # 如果是文件
                if os.path.splitext(newDir)[1] == ".txt":  # 判断是否是txt
                    s = s.split(".")
                    self.readEditFile(newDir, s)  # 读文件
                else:
                    self.eachFile_0(newDir)