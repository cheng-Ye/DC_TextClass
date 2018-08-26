#-*- coding: UTF-8 -*-  
import numpy 
import copy
#import theano
import pickle



def genBatch(data):
    m =0 
    #maxsentencenum = len(data[0])
    maxsentencenum = max([len(item) for item in data])
    for doc in data:
        for sentence in doc:
            if len(sentence)>m:
                m = len(sentence)
        for i in xrange(maxsentencenum - len(doc)):
            doc.append([-1])
    tmp = map(lambda doc: numpy.asarray(map(lambda sentence : sentence + [-1]*(m - len(sentence)), doc), dtype = numpy.int32).T, data)                          #[-1]是加在最前面
    #print [numpy.array(i).shape      for  i in tmp   ]
    tmp = reduce(lambda doc,docs : numpy.concatenate((doc,docs),axis = 1),tmp)
    return tmp 
            
def genLenBatch(lengths,maxsentencenum):
    lengths = map(lambda length : numpy.asarray(length + [1.0]*(maxsentencenum-len(length)), dtype = numpy.float32)+numpy.float32(1e-4),lengths)
    return reduce(lambda x,y : numpy.concatenate((x,y),axis = 0),lengths)

def genwordmask(docsbatch):
    mask = copy.deepcopy(docsbatch)
    mask = map(lambda x : map(lambda y : [1.0 ,0.0][y == -1],x), mask)
    mask = numpy.asarray(mask,dtype=numpy.float32)
    mask[0] = numpy.ones([mask.shape[1]],dtype=numpy.float32) 
    return mask

def gensentencemask(sentencenum):
    #maxnum = sentencenum[0]
    maxnum = max([item for item in sentencenum])
    mask = numpy.asarray(map(lambda num : [1.0]*num + [0.0]*(maxnum - num),sentencenum), dtype = numpy.float32)
    return mask.T

class Dataset(object):
    def __init__(self, filename, emb,maxbatch = 160):
       
        with open(filename,'rb') as f:
            tmp=pickle.load(f)
        print filename,len(tmp)

        docs, label = zip(*tmp)
        sentencenum = map(lambda x : len(x),docs)
        length = map(lambda doc : map(lambda sentence : len(sentence), doc), docs)
        self.epoch = len(docs) / maxbatch                                        
        if len(docs) % maxbatch != 0:
            self.epoch += 1
        
        self.docs = []
        self.label = []
        self.wordmask = []
        self.sentencemask = []
        self.maxsentencenum = []

        for i in xrange(self.epoch):
            self.maxsentencenum.append(max(sentencenum[i*maxbatch:(i+1)*maxbatch]))
            docsbatch = genBatch(docs[i*maxbatch:(i+1)*maxbatch])
            self.docs.append(docsbatch)
            self.label.append(numpy.asarray(label[i*maxbatch:(i+1)*maxbatch], dtype = numpy.int32))
            self.wordmask.append(genwordmask(docsbatch))
            self.sentencemask.append(gensentencemask(sentencenum[i*maxbatch:(i+1)*maxbatch]))
        

class Wordlist(object):
    def __init__(self, filename):
        with open(filename,'rb') as f:
            self.voc =pickle.load(f)
        self.size = len(self.voc )
        #print  self.voc,type(self.voc),self.size

    def getID(self, word):
        try:
            return self.voc[word]
        except:
            return -1

