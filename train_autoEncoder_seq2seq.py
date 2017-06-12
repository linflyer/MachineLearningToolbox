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
from keras.models import Sequential
from keras.optimizers import RMSprop
from seq2seq.models import SimpleSeq2Seq
import pickle

EMBEDDING_DIM = 50

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - one required argument: <data directory>\n")
        sys.exit(-1)
    working_dir = args[0]
    data_file = os.path.join(working_dir, 'thyme_relational_onePathsBetweenArgs.txt')

    sys.stderr.write("loading data file...\n")
    # learn alphabet from training data
    provider = dataset_noLabel.DatasetProvider(data_file)
    # now load training examples and labels
    train_x = provider.load(data_file)

    sys.stderr.write("finished loading data file...\n")
    # turn x into numpy array among other things
    maxlen = max([len(seq) for seq in train_x])
    train_x =[x[0:maxlen] for x in train_x]
    train_x = pad_sequences(train_x, maxlen=maxlen)

    # prepare embedding matrix
    nb_words = len(provider.word2int)

    train_y = np.zeros((train_x.shape[0], maxlen, nb_words), dtype=np.bool)
    inst = 0
    for x in train_x:
        train_y[inst, 0, x[0]]=1
        train_y[inst, 1, x[1]]=1
        inst=inst+1



    pickle.dump(maxlen, open(os.path.join(working_dir, 's2s_maxlen.p'),"wb"))
    pickle.dump(provider.word2int, open(os.path.join(working_dir, 's2s_word2int.p'),"wb"))
    #pickle.dump(nb_words, open(os.path.join(working_dir, 'wd_nb_words.p'),"wb"))
    #pickle.dump(train_x, open(os.path.join(working_dir, 'train_x.p'),"wb"))

    sys.stderr.write("training encoder...\n")

    model = Sequential()
    model.add(Embedding(len(provider.word2int),
             EMBEDDING_DIM,
             input_length=maxlen,
             weights=None))
    seq2seq = SimpleSeq2Seq(
        input_dim=EMBEDDING_DIM,
        input_length=maxlen,
        hidden_dim=10,
        output_dim=nb_words,
        output_length=maxlen,
        depth=1)
    model.add(seq2seq)

    model.compile(optimizer='RMSprop', loss='mse')
    model.fit(train_x,train_y, batch_size=50, nb_epoch=3)

    json_string = model.to_json()
    open(os.path.join(working_dir, 's2s_encoder_0.json'), 'w').write(json_string)
    model.save_weights(os.path.join(working_dir, 's2s_encoder_0.h5'), overwrite=True)

if __name__ == "__main__":
    main(sys.argv[1:])

