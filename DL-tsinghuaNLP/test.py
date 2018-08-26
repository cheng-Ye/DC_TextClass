#-*- coding: UTF-8 -*-  
import sys
from Dataset import *
from LSTMModel import LSTMModel

classes=19
voc = Wordlist('file//voc.pkl')

trainset=None
#trainset = Dataset('data\\'+dataname+'\\train.txt', voc )
testset = Dataset('file//testset.pkl', voc)
print 'data loaded.'


model = LSTMModel(voc.size, trainset, devset,classes,'model')
print 'model loaded.'

preds, label,_=model.test()
