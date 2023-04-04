# -*- coding: utf-8 -*-
"""
Created on Sun May  5 10:51:31 2019

@author: Administrator
"""
import pandas as pd
import numpy as np
import jieba
data1 = pd.read_excel(r'E:\硕士-毕业论文\2019-需求聚类分析\LDA\LDA-需求分析数据样本.xlsx',sheet_name="1-递送速度",encoding='utf-8').astype(str)
data2 = pd.read_excel(r'E:\硕士-毕业论文\2019-需求聚类分析\LDA\LDA-需求分析数据样本.xlsx',sheet_name="2-服务人员素质",encoding='utf-8').astype(str)
data3 = pd.read_excel(r'E:\硕士-毕业论文\2019-需求聚类分析\LDA\LDA-需求分析数据样本.xlsx',sheet_name="3-货物完好程度",encoding='utf-8').astype(str)
data4 = pd.read_excel(r'E:\硕士-毕业论文\2019-需求聚类分析\LDA\LDA-需求分析数据样本.xlsx',sheet_name="4-快递价格",encoding='utf-8').astype(str)
data5 = pd.read_excel(r'E:\硕士-毕业论文\2019-需求聚类分析\LDA\LDA-需求分析数据样本.xlsx',sheet_name="5-快件信息",encoding='utf-8').astype(str)
data6 = pd.read_excel(r'E:\硕士-毕业论文\2019-需求聚类分析\LDA\LDA-需求分析数据样本.xlsx',sheet_name="6-快递预约准时",encoding='utf-8').astype(str)
'''
data1 = pd.read_excel(r'E:\硕士-毕业论文\2019-需求聚类分析\LDA\各公司数据预处理后-2MY-LDA.xlsx',sheet_name="1-递送速度",encoding='utf-8').astype(str)
data2 = pd.read_excel(r'E:\硕士-毕业论文\2019-需求聚类分析\LDA\各公司数据预处理后-2MY-LDA.xlsx',sheet_name="2-服务人员素质",encoding='utf-8').astype(str)
data3 = pd.read_excel(r'E:\硕士-毕业论文\2019-需求聚类分析\LDA\各公司数据预处理后-2MY-LDA.xlsx',sheet_name="3-货物完好程度",encoding='utf-8').astype(str)
data4 = pd.read_excel(r'E:\硕士-毕业论文\2019-需求聚类分析\LDA\各公司数据预处理后-2MY-LDA.xlsx',sheet_name="4-快递价格",encoding='utf-8').astype(str)
data5 = pd.read_excel(r'E:\硕士-毕业论文\2019-需求聚类分析\LDA\各公司数据预处理后-2MY-LDA.xlsx',sheet_name="5-快件信息",encoding='utf-8').astype(str)
data6 = pd.read_excel(r'E:\硕士-毕业论文\2019-需求聚类分析\LDA\各公司数据预处理后-2MY-LDA.xlsx',sheet_name="6-快递预约准时",encoding='utf-8').astype(str)
'''
stopwords_path = r'E:\硕士-毕业论文\2019-需求聚类分析\LDA\哈工大停用词表.txt' # 停用词词表
X_csv_all = []
X_labeled_all = []
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8-sig').readlines()]
    return stopwords
def cutstopwords(review):
    stopwords = stopwordslist(stopwords_path)
    words = [word for word in review if not word in stopwords if word!=' ']
    return(words)
def cutreviews_csv (review): 
    seg_list = jieba.cut(review.replace(' ',''), cut_all=False)
    #seg_list = jieba.cut(review, cut_all=False)
    cut_reviews=cutstopwords(seg_list)
    return " ".join(cut_reviews)
def cutreviews (review): 
    seg_list = jieba.cut(review.replace(' ',''), cut_all=False)
    #seg_list = jieba.cut(review, cut_all=False)
    cut_reviews=cutstopwords(seg_list)
    return cut_reviews
def mergeSeg(data):
    for review in data["评论"]:
        #X_csv_all.append(cutreviews_csv(review))
        X_labeled_all.append(cutreviews(review))
def test_merge():
    print("Merging ...")
    mergeSeg(data1)
    mergeSeg(data2)
    mergeSeg(data3)
    mergeSeg(data4)
    mergeSeg(data5)
    mergeSeg(data6)
    #ALL_SEG_Result = pd.DataFrame(data=X_csv_all, columns=['seg_lists'])
    #ALL_SEG_Result.to_csv('ALL_SEG_Result2.csv', index=False,encoding="utf_8_sig")
    print("Merging end...")

test_merge()
    
juCounter=len(data1)+len(data2)+len(data3)+len(data4)+len(data5)+len(data6)
print("语料库包含",juCounter,"条评论")

from gensim.models import ldamodel,TfidfModel
from gensim.corpora import Dictionary
dictionary = Dictionary(X_labeled_all)
print("词汇表长度为",len(dictionary))
corpus = [dictionary.doc2bow(text) for text in X_labeled_all]
ciCounter=sum(cnt for document in corpus for _, cnt in document)
print("语料词语总数为",ciCounter)

def model_lda(corpus,num):
    print("构建LDA...")
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lda = ldamodel.LdaModel(corpus=corpus_tfidf,id2word=dictionary,num_topics=num,passes=10)
    #model_name = "lda",num
    #lda.save(model_name)
    #lda = ldamodel.LdaModel.load(model_name)
    print("构建LDA end...")
    return lda
def cal_perplexity(lda,corpus):
    print("计算LDA模型的困惑值...")
    #计算方法一
    #perplex=lda.bound(corpus)
    #print("似然值和为",perplex)
    #print("LDA模型的困惑值",np.exp2(-perplex/ciCounter))
    #计算方法二
    perplexity=lda.log_perplexity(corpus)
    print("每个单词的似然值为",perplexity)
    print("LDA模型的困惑值",np.exp2(-perplexity))
    print("计算LDA模型的困惑值 end...")
    return np.exp2(-perplexity)
    
lda=model_lda(corpus,6)
perplexity=cal_perplexity(lda,corpus)

import pyLDAvis.gensim
def test_lda(lda, corpus, dictionary):
    data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    pyLDAvis.show(data,open_browser=True)
    print("LDA end...")
 
test_lda(lda, corpus, dictionary)
