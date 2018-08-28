#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : sparl_test1.py
# @Author: WangYe
# @Date  : 2018/8/14
# @Software: PyCharm
import pyspark
from pyspark import SparkContext
from pyspark import SparkConf
import os
os.environ['JAVA_HOME']='D:\Java\jdk1.8'
conf=SparkConf().setAppName("miniProject").setMaster("local[*]")
sc=SparkContext.getOrCreate(conf)

#（a）利用list创建一个RDD;使用sc.parallelize可以把Python list，NumPy array或者Pandas Series,Pandas DataFrame转成Spark RDD。
rdd = sc.parallelize([1,2,3,4,5])
#rdd
#Output:ParallelCollectionRDD[0] at parallelize at PythonRDD.scala:480

#（b）getNumPartitions()方法查看list被分成了几部分
rdd.getNumPartitions()
#Output:4