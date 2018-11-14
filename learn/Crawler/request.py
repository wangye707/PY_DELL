#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : request.py
# @Author: WangYe
# @Date  : 2018/11/6
# @Software: PyCharm
# coding = utf-8
import urllib.request
import re
import requests


def getDatas(keyword, pages):
    params = []
    for i in range(30, 30 * pages + 30, 30):
        params.append({
            'tn': 'resultjson_com',
            'ipn': 'rj',
            'ct': 201326592,
            'is': '',
            'fp': 'result',
            'queryWord': keyword,
            'cl': 2,
            'lm': -1,
            'ie': 'utf-8',
            'oe': 'utf-8',
            'adpicid': '',
            'st': -1,
            'z': '',
            'ic': 0,
            'word': keyword,
            's': '',
            'se': '',
            'tab': '',
            'width': '',
            'height': '',
            'face': 0,
            'istype': 2,
            'qc': '',
            'nc': 1,
            'fr': '',
            'pn': i,
            'rn': 30,
            'gsm': '1e',
            '1526377465547': ''
        })
    url = 'http://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&word=%E8%B5%B5%E4%B8%BD%E9%A2%96'
    urls = []
    for i in params:
        urls.append(requests.get(url, params=i).json().get('data'))

    return urls


def getImg(datalist, path):
    x = 0
    for list in datalist:
        for i in list:
            if i.get('thumbURL') != None:
                print('正在下载：%s' % i.get('thumbURL'))
                urllib.request.urlretrieve(i.get('thumbURL'), path + '%d.jpg' % x)
                x += 1
            else:
                print('图片链接不存在')


if __name__ == '__main__':
    datalist = getDatas('高清电脑背景', 1)
    getImg(datalist, r'C:\Users\wy\Desktop\图片/')