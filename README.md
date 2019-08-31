# 智慧中国杯之达观数据 

赛题介绍：建立模型通过长文本数据正文(article)，预测文本对应的类别(class)   链接：

思路：建立两个模型，第一个自己搭的，基于ML方法,提取lda,tfidf等特征，离群检测、特征选择，stacking,voting模型；
第二个建立 DL 模型：利用 gensim 构建文本 word2vec 词向量，采用 CNN、GRU、LSTM、attention 等方法预测；使用 fasttext 方法预测；


model1:  所需环境：anaconda3.5,python3.6,gensim,nltk,sklearn，mlxtend

model2: 所需环境：anaconda3.5,python2.7,theano==1.0,sklearn,gensim



