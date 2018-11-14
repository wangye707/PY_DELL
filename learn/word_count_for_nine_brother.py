#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : word_count_for_nine_brother.py
# @Author: WangYe
# @Date  : 2018/9/14
# @Software: PyCharm

def wordcount(path):
    wordcounts={}
    with open(path) as f:
        content = f.read().strip().split()
        for word in content:
            word = word.rstrip('.').rstrip(',').rstrip(':').strip("()")  # 去除单词前后的标点符号
            if word not in wordcounts:
                wordcounts[word] = 1
            else:
                wordcounts[word] += 1
    print(wordcounts)
if __name__ == '__main__':

    path=r'C:\Users\wy\Desktop\data\gettysburg.txt'
    wordcount(path=path)