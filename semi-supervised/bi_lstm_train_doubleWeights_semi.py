#!/usr/bin/env python

import sklearn as sk
import numpy as np
np.random.seed(1337)
from sklearn.metrics import f1_score
import sys
import os.path
import dataset_semi
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input,LSTM,concatenate, RepeatVector, GaussianNoise
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, Callback
from keras.initializers import RandomUniform, VarianceScaling
from keras import backend as K
from keras import regularizers
import pickle

class Metrics(Callback):
    def on_epoch_end(self, batch, logs={}):
        predict = np.asarray(self.model.predict(self.validation_data[0]))
        targ = self.validation_data[1]
        self.f1s=f1_score(np.argmax(targ, axis=1), np.argmax(predict,axis=1), average='weighted')
        return

def customized_catecros(y_true, y_pred):
    return K.mean(K.switch(K.greater(y_true,0), K.square(y_pred - y_true), 0), axis=-1) #K.categorical_crossentropy(y_true, y_pred)

def to_categorical(y, num_classes):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    # if not num_classes:
    #     num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    #categorical[np.arange(n), y] = 1
    for i in range(n):
        if y[i]< num_classes:
            categorical[i, y[i]] = 1
    return categorical

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - one required argument: <data directory>\n")
        sys.exit(-1)
    working_dir = args[0]
    data_file = os.path.join(working_dir, 'training-data.liblinear')

    # learn alphabet from training data
    provider = dataset_semi.DatasetProvider(data_file)
    # now load training examples and labels
    train_x, train_y = provider.loadOne(data_file)
    # turn x and y into numpy array among other things
    #maxlen = max([len(seq) for seq in train_x+unlabeled_x])
    l=[len(seq) for seq in train_x]
    maxlen = int(np.rint(np.mean(l) + 2* np.std(l)))
    train_x =[x[0:maxlen] for x in train_x]
    classes = len(set(train_y))-1

    train_x = pad_sequences(train_x, maxlen=maxlen)
    train_y = to_categorical(np.array(train_y), classes)

    #loading pre-trained embedding file:
    embeddings_index = {}
    f = open(os.path.join(working_dir, 'mt_timex_w2v.vec'))#mimic.txt'))
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
    embedding_matrix1 = np.zeros((nb_words, EMBEDDING_DIM))
    ivw_num = 0
    print('MIMIC embedding OOV words are:')
    for word, i in provider.word2int.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:# words not found in embedding index will be all-zeros.
            embedding_matrix1[i] = embedding_vector
	    ivw_num += 1
	#else:
	    #print('%s' % word)
	    #embedding_matrix1[i] = embeddings_index.get('oov')
    print('%s words out of %s words are in vocabulary.' % (ivw_num, nb_words))

    #loading pre-trained embedding file:
    embeddings_index = {}
    f = open(os.path.join(working_dir, 'tr_fast.vec'))
    values = f.readline().split()
    EMBEDDING_WORDNUM2 = int(values[0])
    EMBEDDING_DIM2=int(values[1])
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('load embeddings for %s=%s words.' % (len(embeddings_index),EMBEDDING_WORDNUM2))

    # prepare embedding matrix
    embedding_matrix2 = np.zeros((nb_words, EMBEDDING_DIM2))
    ivw_num = 0
    print('TR embedding OOV words are:')
    for word, i in provider.word2int.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:# words not found in embedding index will be all-zeros.
            embedding_matrix2[i] = embedding_vector
	    ivw_num += 1
	#else:
	#    print('%s' % word)
	#     embedding_matrix2[i] = embeddings_index.get('oov')
    print('%s words out of %s words are in vocabulary.' % (ivw_num, nb_words))

    pickle.dump(maxlen, open(os.path.join(working_dir, 'maxlen.p'),"wb"))
    pickle.dump(provider.word2int, open(os.path.join(working_dir, 'word2int.p'),"wb"))
    pickle.dump(provider.label2int, open(os.path.join(working_dir, 'label2int.p'),"wb"))

    print 'train_x shape:', train_x.shape
    print 'train_y shape:', train_y.shape

    LSTM_DIM=512
    DROPOUT =0.5

    input = Input(shape=(maxlen,), dtype='int32')
    embed1  = Embedding(nb_words,
                EMBEDDING_DIM,
                #mask_zero=True,
                input_length=maxlen,
                weights=[embedding_matrix1],
		        embeddings_regularizer=regularizers.l2(0.0001),
                trainable=True)(input)
    embed2  = Embedding(nb_words,
                EMBEDDING_DIM2,
                #mask_zero=True,
                input_length=maxlen,
                weights=[embedding_matrix2],
		        embeddings_regularizer=regularizers.l2(0.0001),
                trainable=True)(input)
    embed = concatenate([embed1,embed2])
    lstm_fw = LSTM(LSTM_DIM,
                dropout = DROPOUT,
		#activation='tanh',
                recurrent_dropout = DROPOUT)(embed)
    lstm_bw = LSTM(LSTM_DIM,
                dropout = DROPOUT,
                #activation='tanh',
                recurrent_dropout = DROPOUT,
                go_backwards=True)(embed)
    cat = concatenate([lstm_fw,lstm_bw])

    out  = Dense(classes, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros')(cat)

    model = Model(inputs=[input], outputs=[out])
    #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    optimizer = Adam(lr=0.001)
    model.compile(loss=customized_catecros,#'categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    stopper = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')
    metrics = Metrics()
    model.fit(train_x,
            train_y,
            epochs=11,
            batch_size=256,
            verbose=1)
            #validation_split=0.1)
            #callbacks=[stopper,metrics])

    #Save partly trained model
    # model.save('partly_trained.h5')
    # del model
    json_string = model.to_json()
    open(os.path.join(working_dir, 'model_0.json'), 'w').write(json_string)
    model.save_weights(os.path.join(working_dir, 'model_0.h5'), overwrite=True)
    sys.exit(0)

if __name__ == "__main__":
    main(sys.argv[1:])
