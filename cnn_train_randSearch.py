#!/usr/bin/env python

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
from sklearn.cross_validation import train_test_split
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import random
from random_search import RandomSearch


batch_sizes = (32, 64, 128, 256, 512)
filter_nums = (64, 128, 200, 300)
#embed_dims = (100, 200, 300)
widths = ("2,3","2,4","2,5","3,4","3,5","4,5","2,3,4", "2,4,5", "2,3,5", "3,4,5","2,3,4,5")
denses = (100, 150, 200, 250, 300)


def get_random_config():
    config = {}
    config['batch_size'] = random.choice(batch_sizes)
    config['num_filters'] = random.choice(filter_nums)
    #config['embed_dim'] = random.choice(embed_dims)
    config['filters'] = random.choice(widths)
    config['dense'] = random.choice(denses)
    return config

def run_one_eval(epochs, config, train_x, train_y, maxlen, vocab_size, num_outputs, embedding_matrix, EMBEDDING_DIM):
    print("Testing with config: %s" % (config) )
    branches = [] # models to be merged
    train_xs = [] # train x for each branch
    inflows  = [] # placeholder for each input
    for filter_len in config['filters'].split(','):
        branch = Input(shape=(maxlen,))
        embed = Embedding(vocab_size,
                             EMBEDDING_DIM,
                             weights=[embedding_matrix],
                             trainable=True)(branch)
        conv = Conv1D(filters=config['num_filters'],
                kernel_size=int(filter_len),
                #W_regularizer=l2(0.000075),
                padding='valid',
                activation='relu',
                dilation_rate=1)(embed)
        pool = MaxPooling1D(pool_size=2)(conv)
        flat = Flatten()(pool)
        branches.append(flat)
        train_xs.append(train_x)
        inflows.append(branch)
    concat = concatenate(branches)
    drop1 = Dropout(0.25)(concat)
    dense = Dense(200, activation='relu')(drop1)

    drop2 = Dropout(0.25)(dense)
    out   = Dense(num_outputs, activation='softmax')(drop2)

    model = Model(inputs=inflows, outputs=out)
    #optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08)
    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    history = model.fit(train_xs,
                train_y,
                epochs=epochs,
                batch_size=config['batch_size'],
                class_weight='auto',
                verbose=2,
                validation_split=0.1,
                callbacks=[stopper])
    #pred_y = model.predict(valid_x)

    return history.history['loss'][-1]

def main(args):
    #np.random.seed(1337) 
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

    print 'train_x shape:', train_x.shape
    print 'train_y shape:', train_y.shape

    #train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1, random_state=18)

    optim = RandomSearch(lambda: get_random_config(), lambda x, y: run_one_eval(x, y, train_x, train_y, maxlen, len(provider.word2int), classes, embedding_matrix, EMBEDDING_DIM) )
    best_config = optim.optimize()

    print("Best config: %s" % best_config)

    sys.exit(0)

if __name__ == "__main__":
    main(sys.argv[1:])
