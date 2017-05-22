#!python

from keras.models import Sequential, model_from_json
import sys
import os.path
import pickle
from keras.preprocessing.sequence import pad_sequences

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - one required argument: <model directory>\n")
        sys.exit(-1)
    working_dir = args[0]

    model_dir = working_dir
    maxlen   = pickle.load(open(os.path.join(model_dir, "maxlen.p"), "rb"))
    maxlenC  = pickle.load(open(os.path.join(model_dir, "maxlenC.p"), "rb"))
    word2int = pickle.load(open(os.path.join(model_dir, "word2int.p"), "rb"))
    cui2int = pickle.load(open(os.path.join(model_dir, "cui2int.p"), "rb"))
    label2int = pickle.load(open(os.path.join(model_dir, "label2int.p"), "rb"))
    model = model_from_json(open(os.path.join(model_dir, "model_0.json")).read())
    model.load_weights(os.path.join(model_dir, "model_0.h5"))

    int2label = {}
    for label, integer in label2int.items():
      int2label[integer] = label

    while True:
        try:
            line = sys.stdin.readline().rstrip()
            if not line:
                break

            text, cuis = line.strip().split('|')

            feats=[]
            for unigram in text.rstrip().split():
                if unigram in word2int:
                    feats.append(word2int[unigram])
                else:
                    # TODO: 'none' is not in vocabulary!
                    feats.append(word2int['oov_word'])

            tags = []
            for tag in cuis.rstrip().split():
                if tag in cui2int:
                    tags.append(cui2int[tag])
                else:
                    tags.append(cui2int['oov_cui'])

            if len(feats) > maxlen:
                feats=feats[0:maxlen]
            if len(tags) > maxlenC:
                tags = tags[0:maxlenC]

            test_x = pad_sequences([feats], maxlen=maxlen)
            test_cui = pad_sequences([tags], maxlen=maxlenC)

            out = model.predict([test_x,test_cui])[0]

        except KeyboardInterrupt:
            sys.stderr.write("Caught keyboard interrupt\n")
            break

        if line == '':
            sys.stderr.write("Encountered empty string so exiting\n")
            break

        out_str = int2label[out.argmax()]
        print out_str
        sys.stdout.flush()

    sys.exit(0)

if __name__ == "__main__":
    main(sys.argv[1:])
