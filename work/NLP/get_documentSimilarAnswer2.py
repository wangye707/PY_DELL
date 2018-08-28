# -*- coding: utf-8 -*-
"""
@author: Peter
"""
# import os
# import docx
import jieba
import jieba.analyse
import numpy as np
from gensim import corpora
from gensim import models, similarities
from pymongo import MongoClient
from time import time


AI_document = MongoClient("localhost").DJ.AI_document
words = list(AI_document.find())
doc = []
for p in words:
    doc.append({
        "content": p["content"],  # list 一个元素一段话
        "title": p["title"]  # 标题
    })
# print(doc)


def load_data():
    # load_data_start = time()
    titles, contents, title_idx = [], [], []
    for d in doc:
        content = d['content']
        contents.extend(content)
        title_idx.extend([len(titles)]*len(content))
        titles.append(d['title'])
    # load_data_end = time()
    # print("load_data cost time:", load_data_end-load_data_start)
    return titles, contents, title_idx


def word_tokenize(titles, contents, stopwords):
    # word_tokenize_start = time()
    title_tokens, content_tokens = [], []
    for title in titles:
        title_token = [w for w in jieba.lcut(title, cut_all=True)]
        title_tokens.append(title_token)

    for content in contents:
        content_token = [w for w in jieba.lcut(content, cut_all=True)]
        content_tokens.append(content_token)

    # word_tokenize_end = time()
    # print("word_tokenize cost time:", word_tokenize_end-word_tokenize_start)
    return title_tokens, content_tokens


def compute_tfidf_sim(question, title_tokens, content_tokens, stopwords):
    # build word-embedding
    # compute_tfidf_sim_start = time()

    dictionary_content = corpora.Dictionary(content_tokens)
    corpus_content = [dictionary_content.doc2bow(paragraph) for paragraph in content_tokens]
    tfidf_content = models.TfidfModel(corpus_content, id2word=dictionary_content)
    index_content = similarities.MatrixSimilarity(tfidf_content[corpus_content])

    dictionary_title = corpora.Dictionary(title_tokens)
    corpus_title = [dictionary_title.doc2bow(paragraph) for paragraph in title_tokens]
    tfidf_title = models.TfidfModel(corpus_title, id2word=dictionary_title)
    index_title = similarities.MatrixSimilarity(tfidf_title[corpus_title])

    question_tokens = [w for w in jieba.lcut(question, cut_all=True) if w not in stopwords]
    bow1 = dictionary_content.doc2bow(question_tokens)
    bow2 = dictionary_title.doc2bow(question_tokens)
    tfidf_bow1 = tfidf_content[bow1]
    tfidf_bow2 = tfidf_title[bow2]
    sim1 = index_content[tfidf_bow1]
    sim2 = index_title[tfidf_bow2]

    # compute_tfidf_sim_end = time()
    # print("compute_tfidf_sim cost time:", compute_tfidf_sim_end-compute_tfidf_sim_start)
    return sim1, sim2


def main(question):
    # stopwords = [line.strip() for line in open('stopwords.txt', 'r', encoding='utf-8').readlines()]
    stopwords = []
    titles, contents, title_idx = load_data()
    title_tokens, content_tokens = word_tokenize(titles, contents, stopwords)
    sim1, sim2 = compute_tfidf_sim(question, title_tokens, content_tokens, stopwords)
    sim = [(s + sim2[title_idx[i]]) / 2 for i, s in enumerate(sim1)]
    scores = -np.sort(-np.array(sim), kind='heapsort')[:20]
    if scores[1] >= 0.267:
        indices = np.argsort(-np.array(sim), kind='heapsort')[:2]
        docs = [contents[ind] for ind in indices]
        return docs
    else:
        return []



# if __name__ == '__main__':
#     start = time()
#     result = main('十六大主题是什么')
#     print(result)
#     end = time()
#     print(end-start)
