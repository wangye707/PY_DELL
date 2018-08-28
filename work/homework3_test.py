#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : homework3.py
# @Author: WangYe
# @Date  : 2018/4/22
# @Software: PyCharm
# 微博文字的类别识别
import jieba
import os
import pickle  # 持久化
from numpy import *
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer  # TF_IDF向量生成类
from sklearn.datasets.base import Bunch   #Bunch和字典结构类似，也是由键值对组成，和字典区别：其键值可以被实例对象当作属性使用
from sklearn.naive_bayes import MultinomialNB  # 多项式贝叶斯算法


def readFile(path):
    with open(path, 'r', errors='ignore') as file:  # 文档中编码有些问题，所有用errors过滤错误
        content = file.read()
        return content


def saveFile(path, result):
    with open(path, 'w', errors='ignore') as file:
        file.write(result)


def segText(inputPath, resultPath):
    fatherLists = os.listdir(inputPath)  # 主目录
    for eachDir in fatherLists:  # 遍历主目录中各个文件夹
        eachPath = inputPath + eachDir + "/"  # 保存主目录中每个文件夹目录，便于遍历二级文件
        each_resultPath = resultPath + eachDir + "/"  # 分词结果文件存入的目录
        if not os.path.exists(each_resultPath):
            os.makedirs(each_resultPath)
        childLists = os.listdir(eachPath)  # 获取每个文件夹中的各个文件
        for eachFile in childLists:  # 遍历每个文件夹中的子文件
            eachPathFile = eachPath + eachFile  # 获得每个文件路径
          #  print(eachFile)
            content = readFile(eachPathFile)  # 调用上面函数读取内容
            # content = str(content)
            result = (str(content)).replace("\r\n", "").strip()  # 删除多余空行与空格
            # result = content.replace("\r\n","").strip()

            cutResult = jieba.cut(result)  # 默认方式分词，分词结果用空格隔开
            saveFile(each_resultPath + eachFile, " ".join(cutResult))  # 调用上面函数保存文件


def bunchSave(inputFile, outputFile):
    catelist = os.listdir(inputFile) #input路径下个子目录，列表格式
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])  #bunch里创建4个成员
    bunch.target_name.extend(catelist)  # 将类别保存到Bunch对象中
    for eachDir in catelist:    #文本分类的每个标签遍历的传给eachdir
        eachPath = inputFile + eachDir + "/"
        fileList = os.listdir(eachPath)
        for eachFile in fileList:  # 二级目录中的每个子文件  即开始遍历txt文本
            fullName = eachPath + eachFile  # 二级目录子文件全路径
            bunch.label.append(eachDir)  # 当前分类标签  54行
            bunch.filenames.append(fullName)  # 保存当前文件的路径
            bunch.contents.append(readFile(fullName).strip())  # 保存文件词向量
            #strip除去收尾的指定字符，默认空格
    with open(outputFile, 'wb') as file_obj:  # 持久化必须用二进制访问模式打开
        pickle.dump(bunch, file_obj)
        #pickle

def readBunch(path):
    with open(path, 'rb') as file:
        bunch = pickle.load(file)
    return bunch


def writeBunch(path, bunchFile):
    with open(path, 'wb') as file:
        pickle.dump(bunchFile, file)


def getStopWord(inputFile):
    stopWordList = readFile(inputFile).splitlines()
    return stopWordList


def getTFIDFMat(inputPath, stopWordList, outputPath):  # 求得TF-IDF向量
    bunch = readBunch(inputPath)
    tfidfspace = Bunch(target_name=bunch.target_name,label=bunch.label, filenames=bunch.filenames, tdm=[],
                       vocabulary={})
    # 初始化向量空间
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5)
    transformer = TfidfTransformer()  # 该类会统计每个词语的TF-IDF权值
    # 文本转化为词频矩阵，单独保存字典文件
    tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
    tfidfspace.vocabulary = vectorizer.vocabulary_
    writeBunch(outputPath, tfidfspace)


def getTestSpace(testSetPath, trainSpacePath, stopWordList, testSpacePath):
    bunch = readBunch(testSetPath)
    # 构建测试集TF-IDF向量空间
    testSpace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                      vocabulary={})
    # 导入训练集的词袋
    trainbunch = readBunch(trainSpacePath)
    # 使用TfidfVectorizer初始化向量空间模型  使用训练集词袋向量
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5,
                                 vocabulary=trainbunch.vocabulary)
    transformer = TfidfTransformer()
    testSpace.tdm = vectorizer.fit_transform(bunch.contents)
    testSpace.vocabulary = trainbunch.vocabulary
    # 持久化
    writeBunch(testSpacePath, testSpace)


def bayesAlgorithm(trainPath, testPath):
    trainSet = readBunch(trainPath)
    testSet = readBunch(testPath)
    clf = MultinomialNB(alpha=0.001).fit(trainSet.tdm, trainSet.label)
    print(shape(trainSet.tdm))
    print(shape(testSet.tdm))
    predicted = clf.predict(testSet.tdm)
    total = len(predicted)
    rate = 0
    for flabel, fileName, expct_cate in zip(testSet.label, testSet.filenames, predicted):
        if flabel != expct_cate:
            rate += 1
            print(fileName, ":实际类别：", flabel, "-->预测类别：", expct_cate)
    print("erroe rate:", float(rate) * 100 / float(total), "%")


# 分词，第一个是分词输入，第二个参数是结果保存的路径
segText("C:/Users/wy/Desktop/data/", "C:/Users/wy/Desktop/segResult/")
bunchSave("C:/Users/wy/Desktop/segResult/", "C:/Users/wy/Desktop/train_set.dat")  # 输入分词，输出分词向量
stopWordList = getStopWord("C:/Users/wy/Desktop/stop/stop.txt")  # 获取停用词
getTFIDFMat("C:/Users/wy/Desktop/train_set.dat", stopWordList, "C:/Users/wy/Desktop/tfidfspace.dat")  # 输入词向量，输出特征空间

# 训练集
segText("C:/Users/wy/Desktop/test1/", "C:/Users/wy/Desktop/test_segResult/")  # 分词
bunchSave("C:/Users/wy/Desktop/test_segResult/", "C:/Users/wy/Desktop/test_set.dat")
getTestSpace("C:/Users/wy/Desktop/test_set.dat", "C:/Users/wy/Desktop/tfidfspace.dat", stopWordList, "C:/Users/wy/Desktop/testspace.dat")
bayesAlgorithm("C:/Users/wy/Desktop/tfidfspace.dat", "C:/Users/wy/Desktop/testspace.dat")
