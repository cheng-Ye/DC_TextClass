from gensim import corpora,models

import pandas as pd  
#from functools import reduce
from  datetime import datetime
#from goto import with_goto
import os
import pickle

class My_dictionary:
	def __init__(self,train,test,no_below,no_above):
		self.train=train
		self.test=test
		self.no_above=no_above
		self.no_below=no_below
		self.dictionary,self.train_segs,self.test_segs=self.mydictionary()

	def mydictionary(self):
		train_segs=[]
		test_segs=[]
		train_y=[]
		i=0
		for train_chunk,test_chunk in zip(self.train,self.test):

			print('current chunk',i)
			i+=1
			train_y.extend(train_chunk['class'].values)

			train_word_segs=[word_seg.split()  for word_seg in train_chunk['word_seg']]
			test_word_segs=[word_seg.split()  for word_seg in test_chunk['word_seg']]
			
			if  i==1:
				dictionary = corpora.Dictionary(train_word_segs)
			else:
				dictionary.add_documents(train_word_segs)

			for train_word_seg,test_word_seg in zip(train_word_segs,test_word_segs):
				train_segs.append(train_word_seg)
				test_segs.append(test_word_seg)

		dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above, keep_n=None)
		dictionary.compactify() # remove gaps in id sequence after words that were removed

		pd.DataFrame(train_y,columns=['class']).to_csv('file\\train_y.csv')
		return dictionary,train_segs,test_segs

def chunks(arr, n):
    return [arr[i:i+n] for i in range(0, len(arr), n)]

def gen_data(train,test,no_below,no_above):
	mydictionary=My_dictionary(train,test,no_below=no_below,no_above=no_above)
	dictionary=mydictionary.dictionary

	train_seg_id=[chunks([i for i in dictionary.doc2idx(train_seg)if i!=-1],20) for train_seg in mydictionary.train_segs]
	with open('file//train_seg_id.pkl','wb') as f:
		pickle.dump(train_seg_id,f)

	test_seg_id=[chunks([i for i in dictionary.doc2idx(test_seg)if i!=-1],20) for test_seg in mydictionary.test_segs]
	with open('file//test_seg_id.pkl','wb') as f:
		pickle.dump(test_seg_id,f)

	with open('file//voc.pkl','wb') as f:
		pickle.dump(dictionary.token2id,f)
		
	dictionary.save('file//daguan.dict')
	#save_load(mode='save',files={'train_corpus':train_corpus,'test_corpus':test_corpus,'dictionary':dictionary})
	#print(len(corpus),corpus[:10])


if __name__ =='__main__':
	now = datetime.now()
	#hyper-parameters
	nrows=3000 
	no_below=int(0.1*nrows)
	no_above=0.5
	num_topics=300

	train=pd.read_csv('..//new_data//train_set.csv',nrows=nrows, chunksize=500,usecols=['word_seg','class'])

	test=pd.read_csv('..//new_data//test_set.csv',nrows=nrows, chunksize=500,usecols=['word_seg'])
	gen_data(train,test,no_below,no_above)

	print((datetime.now()-now))