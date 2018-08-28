#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : Spark_test.py
# @Author: WangYe
# @Date  : 2018/8/10
# @Software: PyCharm
# spark，伪分布式测试
import pymysql
import pyspark
import os
from pyspark import SparkConf, SparkContext
os.environ['JAVA_HOME']='D:\Java\jdk1.8'
conf = SparkConf().setMaster('local[*]').setAppName('word_count')
sc = SparkContext(conf=conf)
d = ['a b c d', 'b c d e', 'c d e f']
d_rdd = sc.parallelize(d)
print(d_rdd.getNumPartitions())
print(d_rdd.glom().collect())
rdd_res = d_rdd.flatMap(lambda x: x.split(' ')).\
    map(lambda word: (word, 1)).reduceByKey(lambda a, b: a+b)
print(rdd_res)
print(rdd_res.collect())