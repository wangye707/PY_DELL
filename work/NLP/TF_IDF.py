#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : TF_IDF.py
# @Author: WangYe
# @Date  : 2018/7/23
# @Software: PyCharm
import numpy as np
import collections
import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer   #词频矩阵
import pymysql
# db = pymysql.connect(host="localhost", user="root",password="123", db="test", port=3306, charset='utf8')
# cur = db.cursor()
def readfile(lujing):
    #path = "C:/Users/wy/Desktop/jieba.txt"
    path=lujing
    a=[]#存放去掉字符的文本
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            produce1 = line.replace('，', '').replace('、', '').replace('？', '').\
            replace('//',' ').replace('/',' ')
            #print(produce1)
            a.append(produce1)#存放最终的处理后的文本
    path1 = "C:/Users/wy/Desktop/quci.txt"
    with open(path1,'w',encoding='utf-8') as f:
        for i in a:
            f.write(str(i))
    # print(a[0])
    # b=a[0]
    # print(b[1])
    #print(a)
    f.close()
    return a
def TF_IDF():
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    '''测试词频矩阵'''
    b=[]
    for cishu in range(4360):
        b.append(readfile("C:/Users/wy/Desktop/jieba.txt")[cishu])
    test_cipin=vectorizer.fit_transform(b)
    #print(test_cipin.toarray)
    #print(test_cipin.toarray())             #获取次品矩阵
    #print(vectorizer.get_feature_names())   #词带将会存储来vectorizer
    '''测试结束'''
    '''测试tf-idf'''
    test_tfidf=transformer.fit_transform(test_cipin)  #if-idf中的输入为已经处理过的词频矩阵
    #print(test_tfidf.toarray())   #输出词频矩阵的IF-IDF值
    #print(test_tfidf.toarray().shape)
    '''测试结束'''
    #print("请输出要查询的内容:\n")

    while(1):
        input_text=input()
        if input_text!=0:
            input_text_jieba=jieba.cut(input_text)
            '''开始处理输入文本，构建对应的词频矩阵'''
            coll=collections.Counter(input_text_jieba)
            new_vectorizer=[]
            for word in vectorizer.get_feature_names():   #原始词频
                new_vectorizer.append(coll[word])   #构建输入的全新词频
            #print(new_vectorizer)
            '''全新词带构建完成'''
            '''原始词频的TF-IDF词频矩阵进行转置'''
            new_tfidf=np.array(test_tfidf.toarray()).T
            #print(new_tfidf)
            #print(new_tfidf.shape)
            '''矩阵相乘'''
            new_vectorizer=np.array(new_vectorizer).reshape(1,len(new_vectorizer))
            #print(new_vectorizer)
            scores=np.dot(new_vectorizer,new_tfidf)
            # print('预测结果是：')
            # print(scores)
            # print(type(scores))
            #print(type(scores))
            new_scores=list(scores[0])#将得分的一维矩阵转换为列表
            #print(new_scores)
            #print(type(new_scores))
            #print(new_scores[9])
            max_location =sorted(enumerate(new_scores), key=lambda x:x[1])#列表坐标排序，转换为元组
            max_location.reverse() #上面默认为从小到大，将他逆序
            final_location=[]

            for i in range(3):    #在元组中找到匹配度最高的三个数的坐标
                print(max_location[i][0])
                print(max_location[i][1])
                final_location.append(max_location[i][0])
            print("最近匹配到：")
            print(final_location)
            out_put=[]
            for i in range(3):
                #print(b[final_location[i]])
                out_put.append(b[final_location[i]])
                #out_put.append(location(final_location)[i]) #逐个输出数据库中的标准问题格式
            print(out_put)
            #return out_put
        else:
            print('结束')
# def location(final_position):
#     '''反馈并且处理最后的坐标'''
#     mysql_insert = 'select name from wy_test where id=%d'
#     for i in range(3):
#         cur.execute(mysql_insert, final_position[i]+1)
#         name_lastout = cur.fetchall()
#         #print(temp)
#         # 提交到数据库执行
#         db.commit()
#     return name_lastout
if __name__ == '__main__':
    #readfile()
    TF_IDF()