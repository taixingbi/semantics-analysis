from gensim.models import KeyedVectors
import numpy as np
np.random.seed(7)

"""
#### word embedding
if word in not vocabulary:
1. Stem word
2. randomly initialize (-0.25,0.25)
"""

def google_news(vocabulary):
    print("word embedding...")
    wvmodel = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    
    embedding_weights = np.array([wvmodel[w] if w in wvmodel\
                                                        else np.random.uniform(-0.25,0.25,wvmodel.vector_size)\
                                                        for w in vocabulary])
    print(embedding_weights.shape)
    print("word embedding loading done")
    return embedding_weights
