#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : homework2.py
# @Author: WangYe
# @Date  : 2018/4/8
# @Software: PyCharm
# 用贝叶斯预测语意并且分类
from numpy import *
import re
import os
strpath='C:/Users/wy/Desktop/test'
os.chdir(strpath)
a=[]
for (root, dirs, files) in os.walk(strpath):  #列出目录下的所有文件和文件名
    for filename in files:
        #print(os.path.join(root,filename))
        pathload=str(os.path.join(root,filename))  #遍历的获取文件名
        print(pathload)
        f=open(pathload,'r')   #不能用os.open
        for filenum in f.readlines():
            filenum = filenum.replace(",", " ")  # 将，改为空格
          #  filenum = filenum.replace("，", " ")  # 将，改为空格
            filenum = filenum.split()  # 分词
            # i=0
            for word in filenum:
                a.append(word)
               # print(a)
            # for words in filenum:
            #     str(words)
            #     a.append(words)
            #     print(a)
#print(a)
k=len(a)
i=0
c=[]
while(k-i):    #获取一个二维列表
    b=[]
    for words in a[i]:
        #print(words)
        b.append(str(words))
    i=i+1
    c.append((b))
print(c)

def loadDataSet():
    # postingList=b
    # print(b)
    # print(len(postingList))
    #print(postingList)
    postingList=c
    classVec = [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1]   #训练集分类
    return postingList, classVec
# 得到不重复的词的列表
def createVocabList(dataSet):
    vocabSet = set([])                          # 创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)      #  创建两个集合的并集，每篇文档返回的新词集合添加到该词集合中
    return list(vocabSet)


#文档分类
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个和词汇表等长的向量（所有元素都为0）
    for word in inputSet:  # 遍历所有的单词
        if word in vocabList:  # 如果出现了词汇表中的单词
            returnVec[vocabList.index(word)] = 1  # 输出的文档向量对应值设为1
        else:
            #print(" %s 不在词库中!" % word)
            n3=1
    return returnVec

postingList, classVec = loadDataSet()
vocabList = createVocabList(postingList)

vec = setOfWords2Vec(vocabList, postingList[0])
#print(vec)

trainMatrix = []
for i in range(len(postingList)):
    vec = setOfWords2Vec(vocabList, postingList[i])
    trainMatrix.append(vec)


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 动作性文档的出现概率
    p0Num = ones(numWords);
    p1Num = ones(numWords)
    p0Denom = 2.0;
    p1Denom = 2.0  # change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # IF是动作性文档
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)  # 每个元素做除法
    p0Vect = log(p0Num / p0Denom)  #
    return p0Vect, p1Vect, pAbusive
p0v, p1v, pAb = trainNB0(trainMatrix, classVec)
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  #元素相乘，判断概率
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        n = '爱情'
        return n
    else:
        K='暴力'
        return K
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    Entry=input('请输入一句话：')
    i=0
    testEntry1=[]
    for word in Entry:
        testEntry1.append(word)
        i=i+1
    testEntry = testEntry1
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, '类别: ', classifyNB(thisDoc, p0V, p1V, pAb))
    # testEntry = ['我','喜','欢','你']
    # thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    # print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    Entry2= input('请输入一句话：')
    k2 = 0
    testEntry2= []
    for word in Entry2:
        testEntry2.append(word)
        k2 = k2+1
    testEntry3 = testEntry2
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry3))
    print(testEntry3, '类别: ', classifyNB(thisDoc, p0V, p1V, pAb))
testingNB()