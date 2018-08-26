from gensim import corpora,models
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.sparse import csr_matrix
from nltk.util import ngrams
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
		self.train_corpus,self.test_corpus=self.mycorpus()

	def get_ngrams(self,word_list, n ):
	    n_grams = ngrams(word_list, n)
	    return [ '_'.join(grams) for grams in n_grams]
	
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
				#word_seg_bigram =self.get_ngrams(word_seg,n=2)
				train_segs.append(train_word_seg)
				#train_segs.append(train_word_seg+self.get_ngrams(train_word_seg,n=2))
				test_segs.append(test_word_seg)
				#test_segs.append(test_word_seg+self.get_ngrams(test_word_seg,n=2))
			

		dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above, keep_n=None)
		dictionary.compactify() # remove gaps in id sequence after words that were removed

		#下面这个列表用来做深度学习
		#word_id=[[i for i in dictionary.doc2idx(train_seg)if i!=-1] for train_seg in train_segs]
		pd.DataFrame(train_y,columns=['class']).to_csv('file\\train_y.csv')
		return dictionary,train_segs,test_segs
	
	def mycorpus(self):
		train_corpus=[]
		test_corpus=[]
		train_corpus.extend([self.dictionary.doc2bow(word_seg)  for word_seg in self.train_segs ])
		test_corpus.extend([self.dictionary.doc2bow(word_seg)  for word_seg in self.test_segs ])
		return train_corpus,test_corpus

def corpustoarray(corpus):
	# 将gensim中的mm表示转化成numpy矩阵表示
	data = []
	rows = []
	cols = []
	line_count = 0
	for line in corpus:
		for elem in line:
			rows.append(line_count)
			cols.append(elem[0])
			data.append(elem[1])
		line_count += 1
	corpus_array = csr_matrix((data,(rows,cols))).toarray()
	return corpus_array

def save_load(mode,models=False,files=False):
	if not os.path.isdir('file'):
		os.mkdir('file')
	if not os.path.isdir('model'):
		os.mkdir('model')
	print( files.keys())
	if mode == 'save' :
		if models :
			if 'tfidf' in models.keys():
				models['tfidf'].save('model\\model.tfidf')
			if 'lsi' in models.keys():
				models['lsi'].save('model\\model.lsi')
			if 'hdp' in models.keys():
				models['hdp'].save('model\\model.hdp')
			if 'rp' in models.keys():
				models['rp'].save('model\\model.rp')
			if 'log' in models.keys():
				models['tfidf'].save('model\\model.log')

		elif files:
			print ('save dict & corpus')
			if 'dictionary' in files.keys():
				files['dictionary'].save('file\\daguan.dict')
			if 'train_corpus' in files.keys():
				with open ('file\\{train_corpus.pkl' ,'wb') as f:
					pickle.dump(files['train_corpus'],f)
			if 'test_corpus' in files.keys():
				with open ('file\\{test_corpus.pkl' ,'wb') as f:
					pickle.dump(files['test_corpus'],f)
	
	elif mode == 'load':
		if models:
			pass
		elif file:
			try:
				dictionary = corpora.Dictionary.load('file\\daguan.dict')
				with open ('file\\corpus.pkl' ,'rb') as f:
					corpus = pickle.load(f)
			except:
				print ( 'file not exsit!')


def gen_data(train,test,no_below,no_above,num_topics=300):
	mydictionary=My_dictionary(train,test,no_below=no_below,no_above=no_above)
	train_corpus=mydictionary.train_corpus
	test_corpus=mydictionary.test_corpus
	dictionary=mydictionary.dictionary

	save_load(mode='save',files={'train_corpus':train_corpus,'test_corpus':test_corpus,'dictionary':dictionary})
	#print(len(corpus),corpus[:10])

	#tfidf
	print('tfidf...')
	tfidf = models.TfidfModel(train_corpus) # 第一步--初始化一个模型
	train_corpus_tfidf=tfidf[train_corpus]
	test_corpus_tfidf=tfidf[test_corpus]          #对整个语料库实施转换
	train_tfidf_array=corpustoarray(train_corpus_tfidf)
	test_tfidf_array=corpustoarray(test_corpus_tfidf)

	pd.DataFrame(train_tfidf_array).to_csv('file\\train_tfidf.csv')
	pd.DataFrame(test_tfidf_array).to_csv('file\\test_tfidf.csv')

	#lsi      200-500的num_topics维度为“金标准”
	print('lsi...')
	lsi = models.LsiModel(train_corpus_tfidf, id2word=dictionary, num_topics=num_topics) # 初始化一个LSI
	train_corpus_lsi = lsi[train_corpus_tfidf] # 在原始语料库上加上双重包装: bow->tfidf->fold-in-lsi
	test_corpus_lsi=lsi[test_corpus]   
	train_lsi_array=corpustoarray(train_corpus_lsi)
	test_lsi_array=corpustoarray(test_corpus_lsi)
	pd.DataFrame(train_lsi_array).to_csv('file\\train_lsi.csv')
	pd.DataFrame(test_lsi_array).to_csv('file\\test_lsi.csv')

	#lsi=models.LsiModel.load('model\\model.lsi')

	#RP
	print('rp...')
	rp = models.RpModel(train_corpus_tfidf, id2word=dictionary, num_topics= num_topics)
	train_corpus_rp = rp[train_corpus_tfidf]
	test_corpus_rp=rp[test_corpus]   
	train_rp_array=corpustoarray(train_corpus_rp)
	test_rp_array=corpustoarray(test_corpus_rp)
	pd.DataFrame(train_rp_array).to_csv('file\\train_rp.csv')
	pd.DataFrame(test_rp_array).to_csv('file\\test_rp.csv')

	'''
	#LDA   2003    LDA最早由Blei, David M.、吴恩达和Jordan, Michael I于2003年提出    有bug
	lda = models.LdaSeqModel(corpus, id2word=dictionary,num_topics=300)
	corpus_lda=lda[corpus]   
	'''
	#HDP    2011   Wang, Paisley, Blei:  http://proceedings.mlr.press/v15/wang11a/wang11a.pdf  
	print('hdp...')
	hdp = models.HdpModel(train_corpus, id2word=dictionary)
	train_corpus_hdp=hdp[train_corpus]  
	test_corpus_hdp=hdp[test_corpus]   
	train_hdp_array=corpustoarray(train_corpus_hdp)
	test_hdp_array=corpustoarray(test_corpus_hdp)
	pd.DataFrame(train_hdp_array).to_csv('file\\train_hdp.csv')
	pd.DataFrame(test_hdp_array).to_csv('file\\test_hdp.csv')

	#Log Entropy Model   2015 
	print('log...')
	log= models.LogEntropyModel(train_corpus) 
	train_corpus_log =log[train_corpus]
	test_corpus_log=log[test_corpus]   
	train_log_array=corpustoarray(train_corpus_log)
	test_log_array=corpustoarray(test_corpus_log)
	pd.DataFrame(train_log_array).to_csv('file\\train_log.csv')
	pd.DataFrame(test_log_array).to_csv('file\\test_log.csv')


if __name__ =='__main__':
	now = datetime.now()
	#hyper-parameters
	nrows=3000 
	no_below=int(0.1*nrows)
	no_above=0.5
	num_topics=300

	train=pd.read_csv('..\\new_data\\train_set.csv',nrows=nrows, chunksize=500,usecols=['word_seg','class'])
	test=pd.read_csv('..\\new_data\\test_set.csv',nrows=nrows, chunksize=500,usecols=['word_seg'])
	gen_data(train,test,no_below,no_above,num_topics)

	print((datetime.now()-now))