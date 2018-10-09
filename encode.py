import pandas as pd
import numpy as np
import itertools
from collections import Counter

def encode_data(df):
    def get_data():
        x_text = [s.split(" ") for s in df['review'].as_matrix()]
        y=df['sentiment'].as_matrix()
        return [x_text, y]

    def build_vocab(sentences):
        word_counts = Counter(itertools.chain(*sentences))
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    def build_input_data(texts, labels, vocabulary):
        x = np.array([ [vocabulary[str(word)] for word in text] for text in texts ] )
        y = np.array(labels)

        return [x, y]    
    
    texts, labels = get_data()
    vocabulary, vocabulary_inv = build_vocab(texts)
    x, y = build_input_data(texts, labels, vocabulary)
    
    return x, y, vocabulary # vocabulary: dictionary
