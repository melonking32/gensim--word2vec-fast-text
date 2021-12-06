# gensim使用--word2vec,fast-text
##### 研究生课程海量数据处理必做作业2-gensim的使用

####Gensim简介
Gensim是⼀款开源的第三⽅Python⼯具包，⽤于从原始的⾮结构化的⽂本中，⽆监督地学习到⽂本隐层的主题向量表达。  
它⽀持包括TF-IDF，LSA，LDA，和word2vec在内的多种主题模型算法，⽀持流式训练，并提供了诸如相似度计算，信息检索等⼀些常⽤任务的API接口

## 一、训练Word2vec词向量模型

    为了让计算机能读懂自然语言，需要把语言转化为数字。  
    最容易想到的方法就是长度等于词汇数的one-hot向量，每个词都被表示成对应位置为1，其余位置全0的向量。但这样实在是太复杂了（指计算复杂），而且根本无法表示出词语之间的联系（词向量完全正交）。  
    所以后来Hinton老爷子提出了分布式表示这个概念，将每个词表示成长度远小于词典长度的向量，Word2vec采用的就是这种分布式表示的词向量。  
    
    Word2vec包含两种模型，一是CNOW模型，通过给定一个中心词周围的词 去预测中心词，来训练模型表示词的能力；另一种模型则是通过中心词去预测周围的词，叫做skip-gram模型（跳词模型）。本次实验主要使用跳词模型。

##### 1.1首先获取数据并处理
  数据使用的是微博评论数据集，包含很多无关内容，对其进行清洗和分词得到训练数据。  
```python
#coding=utf-8
import jieba
import chardet
import re
import pymysql
import csv
from gensim.models import word2vec
from gensim.models import fasttext
import pandas as pd

#通过正则表达式来简单清洗数据
def clean(text):
    #去除正文中@、回复、//转发中的用户名
    text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)
    #去除话题内容
    # text = re.sub(r"#\S+#", "", text)
    #去除转发微博这种词
    text = text.replace("转发微博", "")
    #去除网址
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text) 
    # 去除表情符号
    text = re.sub(r"\[\S+\]", "", text)
    # 合并正文中过多的空格
    text = re.sub(r"\s+", " ", text) 
    return text.strip()

#导入停用词表，放在stopwords中
def stopWordsList(filepath):
    stopwords=[]
    # stopwords = [line.restrip()for line in open(filepath,'r',encoding='utf-8').readlines]
    with open(filepath,'r',encoding='utf-8')as K:
        for line in K:
            line = line.strip()
            stopwords.append(line)
    return stopwords


stopwords = stopWordsList('hit_stopwords.txt')
with open('test.csv','r',encoding='utf-8') as f:
    for line in f:
        # print(line.split(',')[-1])  
        line = clean(line)
        sentence = line.split(',')[:-1]
        result = ''.join(sentence)
        
        # with open('testClean.txt','a',encoding='utf-8') as C:
        #     C.write(result+'\n')
        with open('testJieba2.txt','a',encoding='utf-8') as k:
            #分词
            word = (" ".join(jieba.lcut(result)))
            final = " "
            #停用词过滤
            for wordone in word:
                if wordone not in stopwords:
                    final+= wordone
            k.write(final)
```

##### 1.2 接下来读取训练数据，训练一个word2vec模型，然后使用训练得到的词向量来计算两个词之间的相似度
```python
data = word2vec.Text8Corpus("testJieba2.txt")
#训练语料
path  = open("model.txt",'w',encoding='utf-8')
model = word2vec.Word2Vec(data, hs=1, min_count=1, window=10, size=100)
#保存模型至W2V.model
model.save('W2V.model')
print('和西装语义最接近的几个词:',model.wv.most_similar('西装'))

#读取模型，下次就不用再训练了
#model = word2vec.Word2Vec.load('W2V.model')
#print(model['幸福'])  #幸福的词向量
```
##### 结果：
![image](https://user-images.githubusercontent.com/51854482/144807076-2f8d2f86-ae94-41c1-bd3d-1c4dca7e529f.png)


## 二、训练Word2vec词向量模型
  word2vec模型没有办法对OOV进行处理，所以使用FastText
##### 只需要改一行训练代码
```python
model = fasttext.FastText(data2,  size=4, window=3, min_count=1, iter=10,min_n = 3 , max_n = 6,word_ngrams = 0)
model.save('model.model')
print('和西瓜语义最接近的几个词:',model.wv.most_similar('西瓜'))
```
##### 结果（西瓜是OOV）




