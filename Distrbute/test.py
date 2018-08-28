#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : test.py
# @Author: WangYe
# @Date  : 2018/8/13
# @Software: PyCharm
from pyspark import SparkConf,SparkContext
conf=SparkConf().setMaster('local[*]').setAppName('word_count')
sc = SparkContext(conf=conf)
d = ['a b c d', 'b c d e', 'c d e f']
d_rdd = sc.parallelize(d)
rdd_res = d_rdd.flatMap(lambda x: x.split(' ')).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a+b)
print(rdd_res.collect())

