#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : ml_test.py
# @Author: WangYe
# @Date  : 2018/8/14
# @Software: PyCharm
from pyspark.ml.regression import LinearRegression
# $example off$
import os
import distkeras
from pyspark import SparkConf,SparkContext
from pyspark.sql import SparkSession
os.environ['JAVA_HOME']='D:\Java\jdk1.8'
if __name__ == "__main__":
    conf = SparkConf().setMaster\
        ('local[*]')
    sc = SparkContext(conf=conf)
    spark = SparkSession\
        .builder\
        .appName("LinearRegressionWithElasticNet")\
        .getOrCreate()

    # $example on$
    # Load training data
    training = spark.read.format("libsvm")\
        .load("data/mllib/sample_linear_regression_data.txt")

    lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # Fit the model
    lrModel = lr.fit(training)

    # Print the coefficients and intercept for linear regression
    print("Coefficients: %s" % str(lrModel.coefficients))
    print("Intercept: %s" % str(lrModel.intercept))

    # Summarize the model over the training set and print out some metrics
    trainingSummary = lrModel.summary
    print("numIterations: %d" % trainingSummary.totalIterations)
    print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
    trainingSummary.residuals.show()
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)
    # $example off$

    spark.stop()