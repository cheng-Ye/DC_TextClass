# 智慧中国杯之达观数据 

赛题介绍：建立模型通过长文本数据正文(article)，预测文本对应的类别(class)   链接：

思路：建立三个模型，第一个基于ML方法,提取lda,tfidf等特征；第二个参考Tsinghua University于2016发表的NLP论文：http://www.thunlp.org/~chm/publications/emnlp2016_NSCUPA.pdf  ；第三个参考最近做的一个项目用到的模型，该模型是由Cornell University 于2018.7.20发表，能够做多种任务的一个模型，paper链接：https://arxiv.org/pdf/1806.08730.pdf；

model1:  所需环境：anaconda3.5,python3.6,gensim,nltk,sklearn，mlxtend

model2: 所需环境：anaconda3.5,python2.7,theano==1.0,sklearn,gensim

model3:所需环境：anaconda3.5,python3.6,pytorch=0.3,gensim


