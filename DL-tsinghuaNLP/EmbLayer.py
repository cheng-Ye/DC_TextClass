#-*- coding: UTF-8 -*-  
import theano
import theano.tensor as T
import numpy
import cPickle

class EmbLayer(object):
    def __init__(self, rng, inp, n_voc, dim, name, dataname,prefix=None):
        self.input = inp
        self.name = name

        if prefix == None:
            
            with open ('model/word2vec.model', 'rb') as f:
                model = cPickle.load(f)
            vocab_size=len([word for word in model.wv.vocab])
            W=numpy.zeros((vocab_size+1,dim))
            for index in range(vocab_size):
                W[index]=model[str(index)]
            W=W.astype(numpy.float32)
            '''
            f = file('model//embinit.save', 'rb')
            W = cPickle.load(f)
            f.close()
            '''
            W = theano.shared(value=W, name='E', borrow=True)    
        else:
            print 'load pretrained word2vec..'
            with open ('model/word2vec.save', 'rb') as f:
                W = cPickle.load(f)
        self.W = W

        self.output = self.W[inp.flatten()].reshape((inp.shape[0], inp.shape[1], dim))
        self.params = [self.W]

    def save(self, prefix):
        f = file('model/word2vec.save', 'wb')
        for obj in self.params:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
