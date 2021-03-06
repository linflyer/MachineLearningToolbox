#!/usr/bin/env python

import sklearn as sk
import numpy as np
np.random.seed(1337)
import sys
import os.path
import dataset_hybrid
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input,LSTM,concatenate
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.initializers import RandomUniform
import pickle
import math

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - one required argument: <data directory>\n")
        sys.exit(-1)
    working_dir = args[0]
    data_file = os.path.join(working_dir, 'training-data.liblinear')

    # learn alphabet from training data
    provider = dataset_hybrid.DatasetProvider(data_file)
    # now load training examples and labels
    train_x, train_cui, train_y = provider.load(data_file)
    # turn x and y into numpy array among other things
    maxlen = max([len(seq) for seq in train_x])
    maxlenC = max([len(seq) for seq in train_cui])
    classes = len(set(train_y))

    train_x = pad_sequences(train_x, maxlen=maxlen)
    train_cui = pad_sequences(train_cui, maxlen=maxlenC)
    train_y = to_categorical(np.array(train_y), classes)

    #loading pre-trained embedding file:
    embeddings_index = {}
    f = open(os.path.join(working_dir, 'mimic.txt'))
    values = f.readline().split()
    EMBEDDING_WORDNUM = int(values[0])
    EMBEDDING_DIM=int(values[1])
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('load embeddings for %s=%s words.' % (len(embeddings_index),EMBEDDING_WORDNUM))

    # prepare embedding matrix
    nb_words = len(provider.word2int)
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in provider.word2int.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:# words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    #loading pre-trained cui embedding file:
    embeddings_cui = {}
    f = open(os.path.join(working_dir, 'cui_embed.txt'))
    EMBEDDING_CUINUM = 268
    EMBEDDING_CUIDIM=  50
    for line in f:
        values = line.split()
        word = values[0].lower()
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_cui[word] = coefs
    f.close()
    print('load embeddings for %s=%s CUIs.' % (len(embeddings_cui),EMBEDDING_CUINUM))

    # prepare embedding matrix
    nb_cuis = len(provider.cui2int)
    embedding_cui_matrix = np.zeros((nb_cuis, EMBEDDING_CUIDIM))
    for cui, i in provider.cui2int.items():
        embedding_vector = embeddings_cui.get(cui)
        if embedding_vector is not None:# words not found in embedding index will be all-zeros.
            embedding_cui_matrix[i] = embedding_vector

    pickle.dump(maxlen, open(os.path.join(working_dir, 'maxlen.p'),"wb"))
    pickle.dump(maxlenC, open(os.path.join(working_dir, 'maxlenC.p'),"wb"))
    pickle.dump(provider.word2int, open(os.path.join(working_dir, 'word2int.p'),"wb"))
    pickle.dump(provider.cui2int, open(os.path.join(working_dir, 'cui2int.p'),"wb"))
    pickle.dump(provider.label2int, open(os.path.join(working_dir, 'label2int.p'),"wb"))

    print 'train_x shape:', train_x.shape
    print 'train_cui shape:', train_cui.shape
    print 'train_y shape:', train_y.shape

    LSTM_DIM=512
    DROPOUT =0.5

    input = Input(shape=(maxlen,), dtype='int32')
    embed  = Embedding(nb_words,
                EMBEDDING_DIM,
                mask_zero=True,
                input_length=maxlen,
                weights=[embedding_matrix],
                trainable=True)(input)
    lstm_fw = LSTM(LSTM_DIM,
                dropout = DROPOUT,
                recurrent_dropout = DROPOUT)(embed)
    lstm_bw = LSTM(LSTM_DIM,
                dropout = DROPOUT,
                recurrent_dropout = DROPOUT,
                go_backwards=True)(embed)


    input_cui = Input(shape=(maxlenC,), dtype='int32')
    embed_cui  = Embedding(nb_cuis,
                EMBEDDING_CUIDIM,
                input_length=maxlenC,
                weights=[embedding_cui_matrix],
                trainable=True)(input_cui)
    conv = Conv1D(filters=50,
                kernel_size=2,
                padding='valid',
                activation='relu',
                strides=1)(embed_cui)
    pool = MaxPooling1D(pool_size=2)(conv)
    flat = Flatten()(pool)

    cat = concatenate([lstm_fw,lstm_bw, flat])

    #drop = Dropout(DROPOUT)(cat)
    # minV = -math.sqrt( 6 )/math.sqrt( LSTM_DIM* 2 + classes)
    # maxV = math.sqrt( 6 )/math.sqrt( LSTM_DIM* 2 + classes)
    # randUni = RandomUniform(minval=minV, maxval=maxV, seed=None)
    out  = Dense(classes, activation='softmax')(cat)    #, kernel_initializer=randUni, bias_initializer='zeros')(cat)
    model = Model(inputs=[input, input_cui], outputs=[out])
    #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    model.fit([train_x,train_cui],
            train_y,
            epochs=20,
            batch_size=256,
            verbose=2,
            validation_split=0.1,
            callbacks=[stopper])

    json_string = model.to_json()
    open(os.path.join(working_dir, 'model_0.json'), 'w').write(json_string)
    model.save_weights(os.path.join(working_dir, 'model_0.h5'), overwrite=True)
    sys.exit(0)

if __name__ == "__main__":
    main(sys.argv[1:])
