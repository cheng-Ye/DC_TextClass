import pickle
import numpy as np 
'''
with open('file//voc.pkl','rb') as f:
	train_seg_id=pickle.load(f)

print(len(train_seg_id))
print(train_seg_id)
print(type(train_seg_id))
'''
with open ('model\\word2vec.model', 'rb') as f:
 model = pickle.load(f)



vocab=[word for word in model.wv.vocab]
vocab_size=len(vocab)
#voc={v: k for k, v in dictionary.token2id.items()}
print vocab[:50],vocab_size

ebbedding_matrix=np.zeros((vocab_size+1,200))
print ebbedding_matrix[0]

for index in range(vocab_size):
    ebbedding_matrix[index]=model[str(index)]

with open ('model\\word2vec.save', 'wb') as f:
    pickle.dump(vec,f)

