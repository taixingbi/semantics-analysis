
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from keras.layers import Dense, GlobalAveragePooling1D
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Convolution1D, MaxPooling1D
from keras.layers import Conv1D, concatenate

from keras.models import Sequential, Model
from keras import regularizers

def cnn(vocabulary_size, max_review_length, embedding_matrix):
    embedding_dim= 300

    model = Sequential()
    model.add(Embedding(input_dim= vocabulary_size, 
                        output_dim= 300, 
                        weights=[embedding_matrix],
                        input_length= max_review_length,
                        trainable=False))
    
    model.add(Dropout(.25, input_shape=(max_review_length, embedding_dim)))
    
########### filter ######################################################
    graph_in = Input(shape=(max_review_length, embedding_dim))
    convs = []
    for filter_length in (2,3,4):
        conv = Convolution1D(nb_filter=128,
                             filter_length= filter_length,
                             border_mode='valid',#no zero padding
                             activation='relu',
                             subsample_length=1)(graph_in)
        pool = MaxPooling1D(pool_length=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)
    model.add( Model(input=graph_in, output= concatenate(convs) ) )
#################################################

    model.add(Dense(50))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    
    model.add( Dense(1, kernel_regularizer=regularizers.l2(0.01)) )#0.01
    
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(model.summary())
    return model



def lstm(vocabulary_size, max_review_length, embedding_matrix):
    embedding_vecor_length = 32
    model = Sequential()
    #model.add(Embedding(vocabulary_size, embedding_vecor_length, input_length= max_review_length))
    model.add(Embedding(input_dim= vocabulary_size, 
                        output_dim= 300, 
                        weights=[embedding_matrix],
                        input_length= max_review_length,
                        trainable=False))    
    
    #model.add(LSTM(100))
    model.add( Bidirectional(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2)))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


