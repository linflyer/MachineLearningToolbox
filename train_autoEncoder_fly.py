#!/usr/bin/env python

import sys
import os.path
import dataset_noLabel
import numpy as np
np.random.seed(1337)

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, RepeatVector
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
import pickle

EMBEDDING_DIM = 50

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - one required argument: <data directory>\n")
        sys.exit(-1)
    working_dir = args[0]
    data_file = os.path.join(working_dir, 'MimicIII_relational_onePathsBetweenArgs.txt')

    sys.stderr.write("loading data file...\n")
    # learn alphabet from training data
    provider = dataset_noLabel.DatasetProvider(data_file)
    # now load training examples and labels
    train_x = provider.load(data_file)

    sys.stderr.write("finished loading data file...\n")
    # turn x into numpy array among other things
    l=[len(seq) for seq in train_x]
    maxlen = int(np.rint(np.mean(l) + 2*np.std(l)))
    train_x =[x[0:maxlen] for x in train_x]
    train_x_rev =[x[::-1] for x in train_x] #reverse its sequence
    train_x_rev = pad_sequences(train_x_rev, maxlen=maxlen)
    train_x = pad_sequences(train_x, maxlen=maxlen, padding='post')

    # prepare output matrix
    nb_words = len(provider.word2int)

    train_y = np.zeros((train_x.shape[0], maxlen, nb_words), dtype=np.float32)#np.bool)
    inst = 0
    for x in train_x:
        for i in range(maxlen):
            train_y[inst, i, x[i]]=1
        inst=inst+1

    pickle.dump(maxlen, open(os.path.join(working_dir, 'path3_rev_maxlen.p'),"wb"))
    pickle.dump(provider.word2int, open(os.path.join(working_dir, 'path3_rev_word2int.p'),"wb"))

    sys.stderr.write("training encoder...\n")
    inputs = Input(shape=(maxlen,), dtype='int32')

    embed  = Embedding(nb_words,
                EMBEDDING_DIM,
                input_length=maxlen,
                weights=None)(inputs)
    #bn0     = BatchNormalization(mode=2)(embed)
    # encoded = LSTM(128,
    #             dropout_W = 0.20,
    #             dropout_U = 0.20,
    #             return_sequences=True)(embed)
    encoded = LSTM(128,
                dropout = 0.20,
                recurrent_dropout = 0.20)(embed)
    decoded = RepeatVector(maxlen)(encoded)
    # decoded = LSTM(128,
    #             dropout_W = 0.20,
    #             dropout_U = 0.20,
    #             return_sequences=True)(decoded)
    decoded = LSTM(nb_words,
                   dropout = 0.20,
                   recurrent_dropout = 0.20,
                   return_sequences=True)(decoded)
    decoded = Activation('softmax')(decoded)

    model = Model(inputs=[inputs], outputs=[decoded])
    encoder = Model(inputs=[inputs],outputs=[encoded])
    optimizer1 = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer1)
    model.fit(train_x_rev,train_y, batch_size=100, epochs=50)

    json_string = encoder.to_json()
    open(os.path.join(working_dir, 'path3_rev_encoder_0.json'), 'w').write(json_string)
    encoder.save_weights(os.path.join(working_dir, 'path3_rev_encoder_0.h5'), overwrite=True)

    json_string = model.to_json()
    open(os.path.join(working_dir, 'path3_rev_model_0.json'), 'w').write(json_string)
    encoder.save_weights(os.path.join(working_dir, 'path3_rev_model_0.h5'), overwrite=True) 

    sys.stderr.write("finished training encoder...\n")
    sys.exit(0)

if __name__ == "__main__":
    main(sys.argv[1:])

