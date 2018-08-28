#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : write_mysql.py
# @Author: WangYe
# @Date  : 2018/7/26
# @Software: PyCharm
import pymysql
from work.NLP.TF_IDF import readfile
import numpy as np
def INSERT_MYSQL():
    db = pymysql.connect(host="localhost", user="root",password="123", db="test", port=3306, charset='utf8')
    cur = db.cursor()
    name_str=readfile("C:/Users/wy/Desktop/jieba.txt")
    content_str=readfile("C:/Users/wy/Desktop/答案.txt")
    temp = []
    for i in range(10):
        temp=[str(name_str[i]),str(content_str[i])]
        #print(temp)
    temp=[' 保持 高度一致', '习近平总书记在在政治立场政治']
    print(temp)
    #insert_name_content=np.array(temp).reshape(10,2)
    mysql_insert = 'insert into wy_test(name,content) values(%s,%s)'
    # for i in range(10):
    #     str1='insert into wy_test(name,content)'
    #     for i in range(10):
    #         str2=str(values(i,name_str[i],content_str[i]))
    #         str3=str1+' '+
    cur.execute(mysql_insert, temp)
    print(temp)
    # 提交到数据库执行
    db.commit()
    try:
        # 执行sql语句
        print(len(temp))
        for i in range(10):
            cur.execute(mysql_insert,temp)
            print(temp)
        # 提交到数据库执行
            db.commit()
    except:
        # 如果发生错误则回滚
        db.rollback()

    db.close()  # 关闭连接

if __name__ == '__main__':
    #show_maysql()
    INSERT_MYSQL()