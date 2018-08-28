# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 16:31:05 2018

@author: Peter
"""
import jieba
import numpy as np
from gensim import corpora
from gensim import models, similarities
def get_most_similar_list(inputQuestion,candQuestionDicts,stopwords_file_path):
	'''
	inputQuestion为用户输入问题，字符串类型
	candQuestions待选问题的一个列表，每一个元素为一个字符串类型
	stopwords_file_path为停用词文件的路径
	'''

	#加载停用词
	stopwords = [line.strip() for line in open(stopwords_file_path, 'r', encoding='utf-8').readlines()]
	
	# [{"sentence": "fdsfdsafdsa", "zid": "abc123"},  {},  {}]
	candQuestions = [candQuestionDict['sentence'] for candQuestionDict in candQuestionDicts]
	zids = [candQuestionDict['zid'] for candQuestionDict in candQuestionDicts]
	candQuestions = [jieba.lcut(candQuestion)for candQuestion in candQuestions]
	#构建词向量
	dictionary = corpora.Dictionary(candQuestions)
	corpus = [dictionary.doc2bow(question) for question in candQuestions]
	tfidf = models.TfidfModel(corpus,id2word = dictionary)
	index = similarities.MatrixSimilarity(tfidf[corpus])
	
	#相似度匹配
	inputQuestion = jieba.lcut(inputQuestion.strip())
	inputQuestion = [t for t in inputQuestion if t not in stopwords]
	inputQuestion = dictionary.doc2bow(inputQuestion)
	te = tfidf[inputQuestion]
	sim = index[te]
	indices = np.argsort(-sim, kind='heapsort')[:5]
	results = [zids[ind] for ind in indices]
	return results

if __name__ == '__main__':
	candQuestionDicts = [{'sentence':'十九大新修订的党章明确，党在社会主义初级阶段的基本路线是什么？','zid':'1'},
				  {'sentence':'新世纪新阶段，经济和社会发展的战略目标是什么？','zid':'2'},
				  {'sentence':'十九大新修订的党章明确，中国特色社会主义事业的战略布局是什么？','zid':'3'},
				  {'sentence':'十九大新修订的党章明确，中国特色社会主义事业的总体布局是什么？','zid':'4'},
				  {'sentence':'十九大新修订的党章明确，必须坚持什么样的发展理念？','zid':'5'},
				  {'sentence':'十九大新修订的党章明确，必须坚持什么样的发展思想？','zid':'6'},
				  {'sentence':'十九大新修订的党章明确了我国社会的主要矛盾是什么？','zid':'7'},
				  {'sentence':'十九大新修订的党章总纲部分提出的全党要为“三个实现”奋斗，指的是什么？','zid':'8'}
				  ]
	inputQuestion = '对于新的阶段，社会发展的目标是什么？'
	results = get_most_similar_list(inputQuestion,candQuestionDicts,'stopwords.txt')