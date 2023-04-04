#coding:utf-8
###sys.setdefaultencoding('utf8')
import sys
import importlib
importlib.reload(sys)
VECTOR_DIR = 'vectors.bin'

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 200
TEST_SPLIT = 0.2



print ('(1) load texts...')
train_texts = open('2-服务人员素质-1500.txt',encoding='gb18030').read().split('\n')
train_labels = open('2-服务人员素质-L1500.txt',encoding='gb18030').read().split('\n')
test_texts = open('2-服务人员素质-500.txt',encoding='gb18030').read().split('\n')
test_labels = open('2-服务人员素质-L500.txt',encoding='gb18030').read().split('\n')
all_text = train_texts + test_texts

print ('(2) doc to var...')
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,CountVectorizer   
count_v0= TfidfVectorizer();  
counts_all = count_v0.fit_transform(all_text);
count_v1= TfidfVectorizer(vocabulary=count_v0.vocabulary_);  
counts_train = count_v1.fit_transform(train_texts);   
print ("the shape of train is "+repr(counts_train.shape))  
count_v2 = TfidfVectorizer(vocabulary=count_v0.vocabulary_);  
counts_test = count_v2.fit_transform(test_texts);  
print ("the shape of test is "+repr(counts_test.shape))

'''
tfidftransformer = TfidfTransformer();    
counts_train = tfidftransformer.fit(counts_train).transform(counts_train);
counts_test = tfidftransformer.fit(counts_test).transform(counts_test); 
'''
#x_train = train_data
#y_train = train_labels
#x_test = test_data
#y_test = test_labels
'''
from sklearn.metrics import precision_score#(混淆矩阵、ROC矩阵)
from sklearn.naive_bayes import MultinomialNB  
clf = MultinomialNB(alpha=1.2)
clf.fit(counts_train, train_labels);  
preds = clf.predict(counts_test);
print(precision_score(test_labels,preds,average='macro'))
'''
print( '(3) SVM...')
 

from sklearn.metrics import precision_score
from sklearn.svm import SVC   
svclf = SVC(kernel = 'linear') 
svclf.fit(counts_train, train_labels)  
preds = svclf.predict(counts_test);
print(precision_score(test_labels,preds,average='macro'))  



num = 0

preds = preds.tolist()
for i,pred in enumerate(preds):
    if int(pred) == int(test_labels[i]):
        num += 1
print("(4) num", num)  
print ('precision_score:' + str(float(num) / len(preds)))

'''

def Pnewtext(filepath):
    num1=0
    num2=0
    with open (filepath,'r',encoding='gb18030') as f1:
        newtext=f1.read().split('\n')
        for i in newtext:
            new_test=count_v0.transform([i])
            newpred=svclf.predict(new_test)
#            print(i)
#            print(newpred)
#            print(i+str(newpred))
#            print(type(newpred))
            if int(newpred)==1:
                with open ('速度1.txt','a',encoding='gb18030') as f2:
                    f2.write(i+str(newpred)+'\n')
                    num1 += 1
#                    print(" num1为"+  format(num1))
            elif int(newpred)==0:
                with open ('速度0.txt','a',encoding='gb18030') as f3:
                    f3.write(i+str(newpred)+'\n')
                    num2 += 1
#                    print(" num2为"+  format(num2))
        print(num1,num2)
Pnewtext('1-递送速度-500 - 副本.txt')                    

'''


        




