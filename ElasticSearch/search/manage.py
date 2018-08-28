from ElasticSearch.search.read import Read
from ElasticSearch.search.update import Update
from sys import argv

import json
filepath = r'F:\rizhi'
red = Read()
red.eachFile(filepath=filepath)
up = Update()
up.update(72)
upfilepath = r'F:\data\ceshi'
red = Read()
red.eachFile_0(filepath=upfilepath)
from .search import SearchEdit
#Search(74)
s=argv[1]
se = SearchEdit()
se.detailSearch(74)
