#!/bin/bash

source $(dirname $0)/../keras/env/bin/activate
python $(dirname $0)/bi_lstm_classify_hybrid.py $*
ret=$?
deactivate
exit $ret

