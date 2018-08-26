from gensim import corpora
from gensim.models import Word2Vec,FastText
import pandas as pd
import numpy as np
import pickle 
import random



def chunks(arr, n):
    tmp=[arr[i:i+n] for i in range(0, len(arr), n)]
    return tmp

def gen_data(train,test,sentence_size,no_below,no_above):
    mydictionary=My_dictionary(train,test,no_below=no_below,no_above=no_above)
    dictionary=mydictionary.dictionary
    
    trainvec=[[str(i)  for i in dictionary.doc2idx(train_seg)if i!=-1] for train_seg in mydictionary.train_segs]
    model= Word2Vec(trainvec,min_count=1,size=200) 
    #model= FastText(mydictionary.train_segs,min_count=1,size=200) 
    print 'save word2vec.model...'
    model.save('model/word2vec.model')
   
    train_seg_id=(chunks([i  for i in dictionary.doc2idx(train_seg)if i!=-1],sentence_size) for train_seg in mydictionary.train_segs)
    
    train_label=pd.read_csv('file/train_label.csv',index_col =0)['class'].values-1

    train_label = np.asarray(train_label,dtype = np.int32 )
    #print 'split trainset and devset...'    
    tmp = zip(train_seg_id, train_label)
    tmp=filter(lambda x:x[0]!=[],tmp)
    random.shuffle(tmp)     
    trainset=tmp[:int(0.8*len(tmp))]
    devset=tmp[-int(0.2*len(tmp)):]
    del tmp  
    trainset.sort(lambda x, y: len(y[0]) - len(x[0]))  
    devset.sort(lambda x, y: len(y[0]) - len(x[0]))
    #docs, label = zip(*tmp)

    #print 'trainset: ',len(trainset),'devset',len(devset)
    #print 'save trainset and devset...'
    with open('file/trainset.pkl','wb') as f:
        pickle.dump(trainset,f)
    with open('file/devset.pkl','wb') as f:
        pickle.dump(devset,f)
  
    #print 'save testset...'  
    test_seg_id=[chunks([i for i in dictionary.doc2idx(test_seg)if i!=-1],sentence_size) for test_seg in mydictionary.test_segs]
    test_label=range(len(test_seg_id))
    testset=zip(test_seg_id,test_label)
    testset.sort(lambda x, y: len(y[0]) - len(x[0]))
    with open('file/testset.pkl','wb') as f:
        pickle.dump(testset,f)

    #id2token={v: k for k, v in dictionary.token2id.items()}
    #print id2token,dictionary.token2id
    with open('file/voc.pkl','wb') as f:
        pickle.dump(dictionary.token2id,f)
        
    dictionary.save('file/daguan.dict')

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

        pd.DataFrame(train_y,columns=['class']).to_csv('file/train_label.csv')
        return dictionary,train_segs,test_segs


if __name__=='__main__':
	nrows=2000
	no_below=5000
	no_above=0.8
	sentence_size=10
	train=pd.read_csv('../new_data/train_set.csv', chunksize=5000,usecols=['word_seg','class'])

	test=pd.read_csv('../new_data/test_set.csv', chunksize=5000,usecols=['word_seg'])
	gen_data(train,test,sentence_size,no_below,no_above)
