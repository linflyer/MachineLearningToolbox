#!/usr/bin/env python

import sklearn as sk
import numpy as np
np.random.seed(1337)
import sys
import os.path
import dataset
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, concatenate
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
import pickle

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - one required argument: <data directory>\n")
        sys.exit(-1)
    working_dir = args[0]
    data_file = os.path.join(working_dir, 'training-data.liblinear')

    # learn alphabet from training data
    provider = dataset.DatasetProvider(data_file)
    # now load training examples and labels
    train_x, train_y = provider.load(data_file)
    # turn x and y into numpy array among other things
    maxlen = max([len(seq) for seq in train_x])
    classes = len(set(train_y))

    train_x = pad_sequences(train_x, maxlen=maxlen)
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

    pickle.dump(maxlen, open(os.path.join(working_dir, 'maxlen.p'),"wb"))
    pickle.dump(provider.word2int, open(os.path.join(working_dir, 'word2int.p'),"wb"))
    pickle.dump(provider.label2int, open(os.path.join(working_dir, 'label2int.p'),"wb"))

    print 'train_x shape:', train_x.shape
    print 'train_y shape:', train_y.shape

    branches = [] # models to be merged
    train_xs = [] # train x for each branch
    inflows  = [] # placeholder for each input

    for filter_len in '2,5'.split(','):
        branch = Input(shape=(maxlen,))
        embed = Embedding(len(provider.word2int),
                             EMBEDDING_DIM,
                             weights=[embedding_matrix],
                             trainable=True)(branch)
        conv = Conv1D(filters=200,
                                 kernel_size=int(filter_len),
                                 padding='valid',
                                 activation='relu',
                                 strides=1)(embed)
        pool = MaxPooling1D(pool_size=2)(conv)
        flat = Flatten()(pool)
        branches.append(flat)
        train_xs.append(train_x)
        inflows.append(branch)

    concat = concatenate(branches)

    drop1 = Dropout(0.25)(concat)
    dense = Dense(200, activation='relu')(drop1)

    drop2 = Dropout(0.25)(dense)
    out   = Dense(classes, activation='softmax')(drop2)

    model = Model(inputs=inflows, outputs=out)
    #optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08)
    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    model.fit(train_xs,
              train_y,
              epochs=20,
              batch_size=128,
              verbose=1,
              validation_split=0.1,
              callbacks=[stopper])

    json_string = model.to_json()
    open(os.path.join(working_dir, 'model_0.json'), 'w').write(json_string)
    model.save_weights(os.path.join(working_dir, 'model_0.h5'), overwrite=True)
    sys.exit(0)

if __name__ == "__main__":
    main(sys.argv[1:])
