#-*- coding: UTF-8 -*-  
import pandas as pd 
from Dataset import *

from LSTMModel import LSTMModel

classes=19
voc = Wordlist('file/voc.pkl')
devset = Dataset('file/devset.pkl', voc )
trainset = Dataset('file/trainset.pkl', voc)
print 'data loaded.'

print '****************************************************************************'

model = LSTMModel(voc.size, trainset, devset,classes,'model')
print '****************************************************************************'
model.train()
print '****************************************************************************'
#print 'test 1'
_,_,f1 = model.test()
model.save('model')
print '****************************************************************************'
print '\n'
for i in xrange(1,98):
	model.train()
	print '****************************************************************************'
	print 'test',i+1
	_,_,newf1=model.test()
	print 'f1:',f1,'newf1:',newf1
    
	print '****************************************************************************'
	print '\n'
	if newf1>f1 :
		f1=newf1
		model.save('model')
print 'bestmodel saved!'

