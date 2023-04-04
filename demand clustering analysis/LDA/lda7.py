
import pandas as pd
import numpy as np
import jieba
'''
data1 = pd.read_excel(r'D:\Documents\WeChat Files\wxid_72w8tr50q1wf22\FileStorage\File\2019-04\各公司数据整理完整 2MY.xlsx',sheet_name="1-递送速度",encoding='utf-8').astype(str)
data2 = pd.read_excel(r'D:\Documents\WeChat Files\wxid_72w8tr50q1wf22\FileStorage\File\2019-04\各公司数据整理完整 2MY.xlsx',sheet_name="2-服务人员素质",encoding='utf-8').astype(str)
data3 = pd.read_excel(r'D:\Documents\WeChat Files\wxid_72w8tr50q1wf22\FileStorage\File\2019-04\各公司数据整理完整 2MY.xlsx',sheet_name="3-货物完好程度",encoding='utf-8').astype(str)
data4 = pd.read_excel(r'D:\Documents\WeChat Files\wxid_72w8tr50q1wf22\FileStorage\File\2019-04\各公司数据整理完整 2MY.xlsx',sheet_name="4-快递价格",encoding='utf-8').astype(str)
data5 = pd.read_excel(r'D:\Documents\WeChat Files\wxid_72w8tr50q1wf22\FileStorage\File\2019-04\各公司数据整理完整 2MY.xlsx',sheet_name="5-快件信息",encoding='utf-8').astype(str)
data6 = pd.read_excel(r'D:\Documents\WeChat Files\wxid_72w8tr50q1wf22\FileStorage\File\2019-04\各公司数据整理完整 2MY.xlsx',sheet_name="6-快递预约准时",encoding='utf-8').astype(str)
'''
data1 = pd.read_excel(r'E:\硕士-毕业论文\2019-需求聚类分析\LDA\各公司数据预处理后-2MY-LDA.xlsx',sheet_name="1-递送速度",encoding='utf-8').astype(str)
data2 = pd.read_excel(r'E:\硕士-毕业论文\2019-需求聚类分析\LDA\各公司数据预处理后-2MY-LDA.xlsx',sheet_name="2-服务人员素质",encoding='utf-8').astype(str)
data3 = pd.read_excel(r'E:\硕士-毕业论文\2019-需求聚类分析\LDA\各公司数据预处理后-2MY-LDA.xlsx',sheet_name="3-货物完好程度",encoding='utf-8').astype(str)
data4 = pd.read_excel(r'E:\硕士-毕业论文\2019-需求聚类分析\LDA\各公司数据预处理后-2MY-LDA.xlsx',sheet_name="4-快递价格",encoding='utf-8').astype(str)
data5 = pd.read_excel(r'E:\硕士-毕业论文\2019-需求聚类分析\LDA\各公司数据预处理后-2MY-LDA.xlsx',sheet_name="5-快件信息",encoding='utf-8').astype(str)
data6 = pd.read_excel(r'E:\硕士-毕业论文\2019-需求聚类分析\LDA\各公司数据预处理后-2MY-LDA.xlsx',sheet_name="6-快递预约准时",encoding='utf-8').astype(str)
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
def cal_perplexity(ldamodel, testset, dictionary, num_topics):
    size_dictionary=len(dictionary.keys())
    perplexity = 0.0
    prob_doc_sum = 0.0
    topic_word_list = [] 
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id, size_dictionary)
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)
    doc_topics_ist = []
    for doc in testset:
        doc_topics_ist.append(ldamodel.get_document_topics(doc, minimum_probability=0))
        #print(ldamodel.get_document_topics(doc, minimum_probability=0))
    testset_word_num = 0
    for i in range(len(testset)):
        prob_doc = 0.0 
        doc = testset[i]
        doc_word_num = 0 
        for word_id, num in doc:
            prob_word = 0.0
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                # p(w) = sum(p(z|d)*p(w|z))
                # p(w)指的是测试集中每个单词出现的概率
                # p(z|d)表示的是一个文档中每个主题出现的概率
                # p(w|z)表示的是词典中的每一个单词在某个主题下出现的概率
                prob_topic = doc_topics_ist[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id][word]
                prob_word += prob_topic*prob_topic_word
            # p(d) = sum(log(p(w)))
            prob_doc += np.log(prob_word) 
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    # perplexity = exp(-sum(p(d)/sum(Nd))
    perplexity = np.exp(-prob_doc_sum/testset_word_num) 
    print ("LDA模型的困惑值为",perplexity)
    return perplexity

import matplotlib.pyplot as plt
def graph_draw(topic,perplexity):  
    print("构建主题数与困惑度的折线图...")           
    x=topic
    y=perplexity
    plt.plot(x,y,color="red",linewidth=2)
    plt.xlabel("Number of Topic")
    plt.ylabel("Perplexity")
    plt.show()
    print("构建主题数与困惑度的折线图 end...")

topic=[]
perplexity_list=[]
for i in range(5,45):
    lda=model_lda(corpus,i)
    perplexity = cal_perplexity(lda, corpus, dictionary, i)
    topic.append(i)
    perplexity_list.append(perplexity)
graph_draw(topic,perplexity_list)

'''
import pyLDAvis.gensim
def test_lda(lda, corpus, dictionary):
    data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    pyLDAvis.show(data,open_browser=True)
    print("LDA end...")
 
test_lda(lda, corpus, dictionary)
'''
