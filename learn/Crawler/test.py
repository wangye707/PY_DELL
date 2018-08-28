#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : test.py
# @Author: WangYe
# @Date  : 2018/5/20
# @Software: PyCharm
import requests
from bs4 import BeautifulSoup
newsurl='https://passport.meituan.com/account/unitivesignup?service=www&continue=http%3A%2F%2Fwww.meituan.com%2Faccount%2Fsettoken%3Fcontinue%3Dhttp%253A%252F%252Fwww.meituan.com%252Faccount%252Fsettings'
res=requests.get(newsurl)
res.encoding='uft-8'
print(res.text)
a='https://passport.meituan.com/account/unitivesignup?service=www&continue=http%3A%2F%2Fwww.meituan.com%2Faccount%2Fsettoken%3Fcontinue%3Dhttp%253A%252F%252Fwww.meituan.com%252Faccount%252Fsettings'